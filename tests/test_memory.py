"""多轮记忆：token 估算、历史清洗、按预算裁剪。"""
import pytest
from langchain_core.runnables import RunnableLambda

from agent.memory import (build_messages, estimate_tokens, _clean_history,
                          _select_recent)


# ----------------------------------------------------------------------
# estimate_tokens：字符启发式，CJK 1:1、其余 4:1 向上取整
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", 0),
    (None, 0),
    ("你好", 2),            # 2 个 CJK
    ("abcd", 1),            # 4 个非 CJK → ceil(4/4)
    ("abcdefgh", 2),
    ("你好ab", 3),          # 2 + ceil(2/4)
    ("，。！", 3),          # 全角标点算 CJK
])
def test_estimate_tokens(text, expected):
    assert estimate_tokens(text) == expected


def test_estimate_tokens_is_monotonic():
    """估算只需是个稳定的预算刻度，但必须单调 —— 更长不可能更便宜。"""
    lengths = [estimate_tokens("字" * n) for n in range(0, 50)]

    assert lengths == sorted(lengths)


# ----------------------------------------------------------------------
# _clean_history：过滤脏数据
# ----------------------------------------------------------------------

def test_clean_history_drops_dirty_entries():
    dirty = [
        {"role": "user", "content": "正常"},
        {"role": "system", "content": "角色不合法"},      # role 非 user/assistant
        {"role": "assistant", "content": ""},             # 空内容
        {"role": "assistant", "content": "   "},          # 全空白
        {"role": "user", "content": None},                # 类型不对
        {"role": "user", "content": ["多模态块"]},         # 非 str
        {"content": "没有 role"},
        {"role": "assistant", "content": "也正常"},
    ]

    assert _clean_history(dirty) == [
        {"role": "user", "content": "正常"},
        {"role": "assistant", "content": "也正常"},
    ]


def test_clean_history_drops_extra_keys():
    """只保留 role/content —— 调用方往消息上挂的旁路字段不能流进模型上下文。"""
    cleaned = _clean_history([{"role": "user", "content": "嗨", "eval": {"x": 1}}])

    assert cleaned == [{"role": "user", "content": "嗨"}]


@pytest.mark.parametrize("value", [None, [], ()])
def test_clean_history_handles_empty(value):
    assert _clean_history(value) == []


# ----------------------------------------------------------------------
# _select_recent：token 预算为主阈值，条数为安全网
# ----------------------------------------------------------------------

def _hist(n, content="字"):
    """构造 n 条历史，默认每条 1 token。"""
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": content}
            for i in range(n)]


def test_select_recent_keeps_everything_when_budget_allows():
    history = _hist(3, "字" * 10)          # 每条 10 token

    kept, overflow = _select_recent(history, max_tokens=100, max_messages=50)

    assert kept == history
    assert overflow == []


def test_select_recent_cuts_oldest_first():
    history = _hist(3, "字" * 10)          # 每条 10 token

    kept, overflow = _select_recent(history, max_tokens=25, max_messages=50)

    assert kept == history[-2:]            # 保留最新的两条
    assert overflow == history[:1]         # 最老的被挤出去


def test_newest_message_is_kept_even_if_it_alone_exceeds_budget():
    """最新一条无条件保留。

    它是最相关的上下文；若因为单条超长就被压进摘要，「最近的对话」反而只剩
    一句二手概括，得不偿失。
    """
    history = [{"role": "user", "content": "字" * 5000}]

    kept, overflow = _select_recent(history, max_tokens=10, max_messages=50)

    assert kept == history
    assert overflow == []


def test_message_count_cap_applies_as_safety_net():
    """大量极短消息时，条数上限兜底（token 预算此时还很宽裕）。"""
    history = _hist(20, "字")

    kept, overflow = _select_recent(history, max_tokens=10_000, max_messages=5)

    assert len(kept) == 5
    assert kept == history[-5:]
    assert len(overflow) == 15


@pytest.mark.parametrize("max_tokens,max_messages", [(0, 50), (2000, 0), (-1, 50)])
def test_thresholds_can_disable_history(max_tokens, max_messages):
    """任一阈值设 0 或负数 = 不携带历史，全部历史都算溢出。"""
    history = _hist(5)

    kept, overflow = _select_recent(history, max_tokens, max_messages)

    assert kept == []
    assert overflow == history


def test_select_recent_on_empty_history():
    assert _select_recent([], 2000, 50) == ([], [])


# ----------------------------------------------------------------------
# build_messages：组装本轮输入
# ----------------------------------------------------------------------

def test_build_messages_appends_query_last():
    result = build_messages("本轮提问", history=None, summary="")

    assert result["messages"] == [{"role": "user", "content": "本轮提问"}]
    assert result["summary"] == ""
    assert result["changed"] is False


def test_build_messages_drops_overflow_when_compression_disabled():
    """关闭压缩时，溢出部分直接丢弃 —— 摘要保持原值，changed 为假。"""
    history = _hist(5, "字" * 50)

    result = build_messages("问", history=history, summary="旧摘要",
                            max_tokens=60, max_messages=50, enabled=False)

    assert result["summary"] == "旧摘要"
    assert result["changed"] is False
    assert result["messages"][-1] == {"role": "user", "content": "问"}


def test_summarize_retreats_to_previous_summary_on_failure(caplog):
    """压缩失败必须退回上一版摘要，而不是让本轮问答中断。

    这里让**模型**抛异常（而不是替换掉 ``_summarize``），走的才是真实的
    容错路径 —— 退路写在 ``_summarize`` 内部的 try/except 里，把它整个换掉
    就等于绕过了被测代码。
    """
    def _boom(_input):
        raise RuntimeError("模型挂了")

    overflowing = _hist(5, "字" * 50)

    with caplog.at_level("WARNING"):
        result = build_messages("问", history=overflowing, summary="旧摘要",
                                max_tokens=60, max_messages=50, enabled=True,
                                model=RunnableLambda(_boom))

    assert result["summary"] == "旧摘要"
    assert result["changed"] is False, "退回原值就不该触发落库"
    assert any("摘要生成失败" in r.message for r in caplog.records)


def test_summarize_without_model_skips_quietly(caplog):
    """没给模型就跳过压缩并退回原值。

    正常路径不会走到（ReactAgent 必传 model），显式挡掉是为了避免拿 None 去拼链，
    在报错里留下一句误导性的「摘要生成失败」。
    """
    overflowing = _hist(5, "字" * 50)

    with caplog.at_level("WARNING"):
        result = build_messages("问", history=overflowing, summary="旧摘要",
                                max_tokens=60, max_messages=50, enabled=True, model=None)

    assert result["summary"] == "旧摘要"
    assert any("未提供模型" in r.message for r in caplog.records)


def test_summarize_merges_overflow_into_summary():
    """成功路径：溢出历史被合并成新摘要，changed 为真以触发落库。"""
    def _merge(_input):
        return "新的合并摘要"

    overflowing = _hist(5, "字" * 50)

    result = build_messages("问", history=overflowing, summary="旧摘要",
                            max_tokens=60, max_messages=50, enabled=True,
                            model=RunnableLambda(_merge))

    assert result["summary"] == "新的合并摘要"
    assert result["changed"] is True
