"""检查项：每项都要覆盖通过 / 未通过 / 不适用三种状态。"""
import pytest

from quality.checks import CheckContext, run_checks
from quality.recommendation import analyze_recommendations
from quality.scenario import detect_scenario

CATEGORY_RESULT = """类别「绘画/音乐」下当前在架可借图书共 2 种：
1. 《西方美术史》作者：张敢，索书号：J110.9/0043，可借 3 本
2. 《聆听音乐》作者：克雷格·莱特，索书号：J605/0021，可借 3 本
"""
FOOTER = "以上推荐书目均已确认在馆且可借，实际库存以到馆时为准。"


def _ctx(query, response, tools=(), reader_id="1005", trace=None):
    """按工具名列表造一个轨迹，再组装 CheckContext。"""
    if trace is None:
        trace = [{"seq": i, "name": name, "args": {}, "result": "", "error": None}
                 for i, name in enumerate(tools)]
    return CheckContext(
        query=query, response=response, trace=trace,
        scenario=detect_scenario(query, trace),
        recommendation=analyze_recommendations(response, trace),
        reader_id=reader_id,
    )


def _by_name(results, name):
    return next(r for r in results if r.name == name)


# ----------------------------------------------------------------------
# 通用项
# ----------------------------------------------------------------------

def test_response_not_empty():
    assert _by_name(run_checks(_ctx("你好", "好的")), "回答非空").passed is True
    assert _by_name(run_checks(_ctx("你好", "   ")), "回答非空").passed is False


def test_no_raw_tool_dump():
    assert _by_name(run_checks(_ctx("你好", "正常回答")), "未直出工具原始返回").passed is True

    bad = _by_name(run_checks(_ctx("你好", "结果：当前在架可借图书共 2 种")),
                   "未直出工具原始返回")
    assert bad.passed is False
    assert "当前在架可借图书共" in bad.note


def test_no_premature_give_up():
    """没查够次数就说「我不知道」属于提前放弃。"""
    early = _by_name(run_checks(_ctx("推荐几本书", "我不知道", tools=["get_user_id"])),
                     "未在预算内空答")
    assert early.passed is False
    assert "1 次" in early.note


def test_give_up_after_full_budget_is_acceptable():
    tools = ["a", "b", "c", "d", "e"]
    ok = _by_name(run_checks(_ctx("推荐几本书", "我不知道", tools=tools)), "未在预算内空答")
    assert ok.passed is True


def test_normal_answer_passes_the_give_up_check():
    assert _by_name(run_checks(_ctx("你好", "图书馆每天开放")), "未在预算内空答").passed is True


# ----------------------------------------------------------------------
# 越权查询
# ----------------------------------------------------------------------

def test_no_unauthorized_read_is_not_applicable_without_borrow_query():
    assert _by_name(run_checks(_ctx("你好", "在的")), "无越权查询").passed is None


def test_no_unauthorized_read_passes_for_own_records():
    trace = [{"seq": 0, "name": "fetch_external_data",
              "args": {"user_id": "1005", "month": "2025-04"}, "result": "{}", "error": None}]
    ctx = _ctx("我的借阅数据", "报告内容", trace=trace)

    assert _by_name(run_checks(ctx), "无越权查询").passed is True


def test_no_unauthorized_read_flags_another_reader():
    """工具层已经拦了，这里再从轨迹上确认一次 —— 失败即安全事件。"""
    trace = [{"seq": 0, "name": "fetch_external_data",
              "args": {"user_id": "1006", "month": "2025-04"}, "result": "{}", "error": None}]
    ctx = _ctx("我的借阅数据", "报告内容", trace=trace)

    result = _by_name(run_checks(ctx), "无越权查询")
    assert result.passed is False
    assert "1006" in result.note


# ----------------------------------------------------------------------
# 书目相关
# ----------------------------------------------------------------------

def test_no_fabricated_books_not_applicable_without_recommendations():
    assert _by_name(run_checks(_ctx("你好", "欢迎")), "无编造书目").passed is None


def test_no_fabricated_books_passes_when_all_from_tool():
    trace = [{"seq": 0, "name": "search_books_by_category", "args": {},
              "result": CATEGORY_RESULT, "error": None}]
    response = "## 推荐书目\n1. 《西方美术史》索书号：J110.9/0043，可借 3 本\n" + FOOTER

    assert _by_name(run_checks(_ctx("推荐书", response, trace=trace)),
                    "无编造书目").passed is True


def test_no_fabricated_books_flags_hallucination():
    trace = [{"seq": 0, "name": "search_books_by_category", "args": {},
              "result": CATEGORY_RESULT, "error": None}]
    response = "## 推荐书目\n《不存在的书》索书号：X1/0001，可借 5 本"

    result = _by_name(run_checks(_ctx("推荐书", response, trace=trace)), "无编造书目")
    assert result.passed is False
    assert "不存在的书" in result.note


def test_call_number_check():
    trace = [{"seq": 0, "name": "search_books_by_category", "args": {},
              "result": CATEGORY_RESULT, "error": None}]

    with_number = "## 推荐书目\n《西方美术史》索书号：J110.9/0043，可借 3 本"
    without_number = "## 推荐书目\n《西方美术史》可在 3 楼找到，可借 3 本"

    assert _by_name(run_checks(_ctx("推荐书", with_number, trace=trace)),
                    "推荐书目均标注索书号").passed is True
    assert _by_name(run_checks(_ctx("推荐书", without_number, trace=trace)),
                    "推荐书目均标注索书号").passed is False


def test_self_contradiction_check():
    trace = [{"seq": 0, "name": "search_books_by_category", "args": {},
              "result": CATEGORY_RESULT, "error": None}]
    response = "## 推荐书目\n《西方美术史》可借 3 本\n注意：《西方美术史》已借空。"

    result = _by_name(run_checks(_ctx("推荐书", response, trace=trace)),
                      "同书未同时推荐与否定")
    assert result.passed is False


# ----------------------------------------------------------------------
# 场景相关
# ----------------------------------------------------------------------

def test_report_chain_check():
    full = ["get_user_id", "get_current_month", "fill_context_for_report",
            "fetch_external_data", "search_books_by_category"]
    partial = ["fill_context_for_report", "fetch_external_data"]

    assert _by_name(run_checks(_ctx("根据我的借阅数据推荐", "报告", tools=full)),
                    "报告流程完整").passed is True

    result = _by_name(run_checks(_ctx("根据我的借阅数据推荐", "报告", tools=partial)),
                      "报告流程完整")
    assert result.passed is False
    assert "search_books_by_category" in result.note


def test_report_chain_check_not_applicable_elsewhere():
    assert _by_name(run_checks(_ctx("有哪些科幻类的书可以借", "有的")),
                    "报告流程完整").passed is None


def test_report_closing_note_check():
    ok = _ctx("根据我的借阅数据推荐", "报告正文\n" + FOOTER,
              tools=["fill_context_for_report", "fetch_external_data",
                     "search_books_by_category"])

    assert _by_name(run_checks(ok), "报告含在馆声明").passed is True

    missing = _ctx("根据我的借阅数据推荐", "报告正文，没有收尾声明",
                   tools=["fill_context_for_report", "fetch_external_data",
                          "search_books_by_category"])
    assert _by_name(run_checks(missing), "报告含在馆声明").passed is False


def test_no_accidental_report_flow():
    """main_prompt.txt 明令：非报告场景绝对不调用 fill_context_for_report。"""
    ok = _by_name(run_checks(_ctx("图书馆有哪些功能分区？", "回答",
                                  tools=["rag_summarize"])),
                  "未误触发报告流程")
    assert ok.passed is True

    bad = _by_name(run_checks(_ctx("图书馆有哪些功能分区？", "回答",
                                   tools=["rag_summarize", "fill_context_for_report"])),
                   "未误触发报告流程")
    assert bad.passed is False
    assert "报告人格" in bad.note


def test_no_accidental_report_flow_not_applicable_in_report_scenario():
    ctx = _ctx("根据我的借阅数据推荐", "报告", tools=["fill_context_for_report"])

    assert _by_name(run_checks(ctx), "未误触发报告流程").passed is None


def test_seat_check():
    assert _by_name(run_checks(_ctx("自习区有座位吗", "有", tools=["check_seat"])),
                    "座位数据来自工具").passed is True
    assert _by_name(run_checks(_ctx("自习区有座位吗", "大概有 30 个")),
                    "座位数据来自工具").passed is False
    assert _by_name(run_checks(_ctx("你好", "在的")), "座位数据来自工具").passed is None


# ----------------------------------------------------------------------
# 框架行为
# ----------------------------------------------------------------------

def test_run_checks_returns_every_check():
    """检查项数量固定，不因场景而变 —— 变的是「判定了几项」。"""
    results = run_checks(_ctx("你好", "在的"))

    assert len(results) == len(run_checks(_ctx("根据我的借阅数据推荐", "报告")))
    assert all(r.name for r in results)


def test_one_broken_check_does_not_kill_the_rest(monkeypatch):
    """单项检查执行失败只标记该项为不适用，不该让整份报告消失。"""
    import quality.checks as checks

    def _boom(_ctx):
        raise ValueError("正则写错了")

    original = checks.ALL_CHECKS
    monkeypatch.setattr(checks, "ALL_CHECKS", (_boom,) + original)

    results = run_checks(_ctx("你好", "在的"))

    broken = _by_name(results, "_boom")
    assert broken.passed is None
    assert "正则写错了" in broken.note
    assert len(results) == len(original) + 1
