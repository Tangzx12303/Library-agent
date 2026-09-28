"""指标计算：分母为 0 一律返回 None（不适用），绝不返回 0。"""
import pytest

from quality.metrics import (compliance_rate, extra_call_rate, f1_score,
                             group_recall, precision, ratio, recall)


def test_ratio_returns_none_on_zero_denominator():
    assert ratio(0, 0) is None
    assert ratio(3, 0) is None
    assert ratio(3, 4) == 0.75


# ----------------------------------------------------------------------
# 准确率 / 召回率
# ----------------------------------------------------------------------

def test_perfect_match():
    predicted, truth = {"a", "b"}, {"a", "b"}

    assert precision(predicted, truth) == 1.0
    assert recall(predicted, truth) == 1.0


def test_partial_overlap():
    predicted, truth = {"a", "b", "c"}, {"a", "b", "d"}

    assert precision(predicted, truth) == pytest.approx(2 / 3)
    assert recall(predicted, truth) == pytest.approx(2 / 3)


def test_empty_prediction_differs_between_precision_and_recall():
    """没做出预测时，两个指标**不对称**，这是刻意的。

    - 准确率不适用：没有推荐就无从判断「推荐的准不准」，返回 0 会让每个知识类
      问题都把管理员的推荐 F1 均值往下拽，问答分布一变趋势图就出现假塌陷。
    - 召回率是 0：真值里有候选书、模型一本没推，这是实实在在的失败。

    这个差别有实际意义 —— 只有当工具确实返回过可借书目、而回答里一本都没推荐
    时，召回率才会是 0。
    """
    assert precision(set(), {"a"}) is None
    assert recall(set(), {"a"}) == 0.0


def test_empty_truth_differs_too():
    """真值为空时同样不对称：召回率不适用，准确率是真错（推荐了不存在的东西）。"""
    assert recall({"a"}, set()) is None
    assert precision({"a"}, set()) == 0.0


def test_complete_miss():
    assert precision({"x"}, {"a", "b"}) == 0.0
    assert recall({"x"}, {"a", "b"}) == 0.0


# ----------------------------------------------------------------------
# F1
# ----------------------------------------------------------------------

@pytest.mark.parametrize("p,r,expected", [
    (1.0, 1.0, 1.0),
    (0.5, 0.5, 0.5),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
])
def test_f1_harmonic_mean(p, r, expected):
    assert f1_score(p, r) == pytest.approx(expected)


def test_f1_propagates_none():
    """任一分量不适用，F1 就不适用。"""
    assert f1_score(None, 0.5) is None
    assert f1_score(0.5, None) is None
    assert f1_score(None, None) is None


# ----------------------------------------------------------------------
# OR 组召回率：不惩罚模型多调同类工具
# ----------------------------------------------------------------------

def test_group_recall_all_satisfied():
    groups = (("fill_context_for_report",), ("search_books_by_category", "search_book"))

    score, unsatisfied = group_recall(
        {"fill_context_for_report", "search_book"}, groups)

    assert score == 1.0
    assert unsatisfied == []


def test_group_recall_or_semantics():
    """组内命中任一即算满足 —— 这是为了不惩罚解释性的额外调用。

    读者问「有哪些科幻类的书可以借」，模型先 search_book 核实再查类别，
    两者同组就不该扣分。
    """
    groups = (("search_books_by_category", "search_book"),)

    assert group_recall({"search_book"}, groups)[0] == 1.0
    assert group_recall({"search_books_by_category"}, groups)[0] == 1.0


def test_group_recall_reports_unsatisfied_groups():
    groups = (("a",), ("b",), ("c",))

    score, unsatisfied = group_recall({"a"}, groups)

    assert score == pytest.approx(1 / 3)
    assert unsatisfied == [("b",), ("c",)]


def test_group_recall_without_groups_is_not_applicable():
    """other / failed 场景没有必需工具 → 不适用，而不是 0 分。"""
    assert group_recall({"anything"}, ()) == (None, [])


# ----------------------------------------------------------------------
# 合规率：基准是禁用集，不是期望集
# ----------------------------------------------------------------------

def test_compliance_is_perfect_when_nothing_forbidden():
    """绝大多数场景禁用集为空 → 恒为 1.0。

    这正是重点：**只惩罚提示词明令禁止的调用**，模型多做的合理解释不算错。
    """
    assert compliance_rate({"rag_summarize", "get_opening_hours"}, set()) == 1.0


def test_compliance_penalizes_forbidden_calls():
    got = compliance_rate({"get_opening_hours", "fill_context_for_report"},
                          {"fill_context_for_report"})

    assert got == 0.5


def test_compliance_is_not_applicable_without_calls():
    assert compliance_rate(set(), set()) is None


def test_extra_call_rate_is_informational():
    """额外调用率只用于展示，不进任何通过率。"""
    assert extra_call_rate({"a", "b", "c"}, {"a"}) == pytest.approx(2 / 3)
    assert extra_call_rate(set(), {"a"}) is None
