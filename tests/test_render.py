"""展示格式化：不适用必须显示成「—」，不能显示成 0%。"""
import pytest

from quality.render import (NOT_APPLICABLE, caption, checks_rows, headline,
                            metric_cards, pass_rate_text, rate, scope_note,
                            state_label, tools_text)


# ----------------------------------------------------------------------
# 比率格式化
# ----------------------------------------------------------------------

def test_rate_renders_none_as_dash():
    """**这是本模块最重要的一条。**

    读者问「图书馆有哪些功能分区」时，推荐准确率是**不适用**。显示成「0%」会让人
    以为系统认为自己答错了 —— 一个纯粹由展示层制造的误会。
    """
    assert rate(None) == NOT_APPLICABLE
    assert rate(None) != "0%"


def test_rate_renders_percentages():
    assert rate(1.0) == "100%"
    assert rate(0.5) == "50%"
    assert rate(2 / 3, digits=1) == "66.7%"
    assert rate(0.0) == "0%", "真的算出来是 0 的时候要如实显示"


def test_pass_rate_text_uses_determined_denominator():
    """分母是「判定过的项数」，不是检查项总数。

    一轮普通问答只判定四五项，剩下的是不适用。写成 4/11 会让读者以为漏了七项。
    """
    assert pass_rate_text(4, 5) == "4/5"
    assert pass_rate_text(0, 0) == NOT_APPLICABLE


# ----------------------------------------------------------------------
# 状态标签：图标 + 文字，不靠颜色单独传达
# ----------------------------------------------------------------------

@pytest.mark.parametrize("state,icon", [("pass", "✅"), ("fail", "❌"), ("na", "➖")])
def test_state_labels_carry_text_not_just_colour(state, icon):
    """色觉障碍用户读不出红绿差异，所以图标必须配文字。"""
    label = state_label(state)

    assert label.startswith(icon)
    assert len(label) > len(icon), "不能只有一个图标"


def test_unknown_state_passes_through():
    assert state_label("weird") == "weird"


# ----------------------------------------------------------------------
# 检查项明细表
# ----------------------------------------------------------------------

COMPACT = {
    "passed": 3, "determined": 5,
    "rec_f1": 0.5, "tool_recall": 1.0,
    "scenario_label": "推荐报告", "model_id": "m", "latency_ms": 1200,
    "tools_called": ["get_user_id", "search_books_by_category"],
    "checks": [
        {"name": "回答非空", "state": "pass", "note": ""},
        {"name": "无编造书目", "state": "fail", "note": "《绘画心理学》不在候选内"},
        {"name": "报告流程完整", "state": "pass", "note": ""},
        {"name": "座位数据来自工具", "state": "na", "note": "非座位场景"},
    ],
}


def test_checks_rows_put_failures_first():
    """未通过的排最前，读者一眼看到问题在哪。"""
    rows = checks_rows(COMPACT)

    assert rows[0]["检查项"] == "无编造书目"
    assert rows[0]["结果"] == "❌ 未通过"


def test_checks_rows_never_produce_none_cells():
    """Arrow 不接受一列里既有 str 又有 None（项目已栽过一次 ArrowTypeError）。"""
    rows = checks_rows({"checks": [{"name": "x", "state": "pass"}]})

    assert rows[0]["说明"] == ""
    assert all(isinstance(v, str) for v in rows[0].values())


def test_checks_rows_on_empty_report():
    assert checks_rows({}) == []
    assert checks_rows(None) == []


def test_checks_rows_do_not_leak_sort_key():
    for row in checks_rows(COMPACT):
        assert set(row) == {"检查项", "结果", "说明"}


# ----------------------------------------------------------------------
# 指标卡
# ----------------------------------------------------------------------

def test_metric_cards_show_three_metrics():
    cards = metric_cards(COMPACT)

    assert [c[0] for c in cards] == ["检查项通过", "推荐 F1", "工具召回"]
    assert cards[0][1] == "3/5"
    assert cards[1][1] == "50%"
    assert cards[2][1] == "100%"


def test_metric_cards_render_dashes_for_not_applicable():
    cards = metric_cards({"passed": 4, "determined": 4, "checks": []})

    assert cards[1][1] == NOT_APPLICABLE
    assert cards[2][1] == NOT_APPLICABLE


def test_metric_cards_for_crashed_turn():
    """崩溃轮不显示比率 —— 没有比率可言，显示 0% 是误导。"""
    cards = metric_cards({"error": "OperationalError: 连接失败",
                          "passed": 0, "determined": 1})

    assert cards[0][1] == "0/1"
    assert cards[1][1] == NOT_APPLICABLE
    assert cards[2][1] == NOT_APPLICABLE
    assert "未正常完成" in cards[0][2]


# ----------------------------------------------------------------------
# 标题与说明
# ----------------------------------------------------------------------

def test_headline_summarises_applicable_metrics_only():
    text = headline(COMPACT)

    assert "3/5 项通过" in text
    assert "推荐 F1 50%" in text
    assert "工具召回 100%" in text


def test_headline_omits_inapplicable_metrics():
    text = headline({"passed": 4, "determined": 4, "checks": []})

    assert text == "4/4 项通过"


def test_headline_for_crashed_turn():
    assert headline({"error": "炸了"}) == "本轮对话异常中断"


def test_caption_mentions_not_applicable_count():
    """说清「—」的含义，否则读者会以为指标缺失是 bug。"""
    text = caption(COMPACT)

    assert "不适用" in text
    assert "1 项" in text          # COMPACT 里有一项 na


def test_tools_text_joins_in_call_order():
    assert tools_text(COMPACT) == "get_user_id → search_books_by_category"
    assert tools_text({}) == "（本轮未调用工具）"


def test_scope_note_states_the_caliber():
    """展示数字的地方就该说清数字管什么 —— 否则会被当成模型的整体评分。"""
    note = scope_note()

    assert "推荐准确率" in note
    assert "推荐召回率" in note
    assert "不惩罚" in note and "额外调用" in note
