"""看板聚合：小样本保护、不适用不参与均值、口径正确。"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from quality.aggregate import (MIN_SAMPLES_FOR_RANKING, by_model,
                               check_failure_rates, daily_trend, kpi_summary,
                               other_share, scenario_check_matrix, to_frame,
                               top_unsatisfied_groups)


def _row(scenario="report", model="m1", passed=3, determined=4,
         rec_f1=None, tool_recall=1.0, source="keyword", day=0,
         checks=None, unsatisfied=None):
    return {
        "id": 1, "reader_id": "1005", "model_id": model, "query": "q",
        "scenario": scenario, "scenario_source": source,
        "passed": passed, "determined": determined,
        "rec_precision": rec_f1, "rec_recall": rec_f1, "rec_f1": rec_f1,
        "tool_precision": tool_recall, "tool_recall": tool_recall, "tool_f1": tool_recall,
        "latency_ms": 1000,
        "created_at": datetime(2026, 9, 20) + timedelta(days=day, hours=1),
        "detail": {"checks": checks or [], "unsatisfied_groups": unsatisfied or [],
                   "tools_called": []},
    }


def _checks(*spec):
    return [{"name": name, "state": state, "note": ""} for name, state in spec]


# ----------------------------------------------------------------------
# to_frame
# ----------------------------------------------------------------------

def test_to_frame_on_empty_input():
    assert to_frame([]).empty


def test_to_frame_coerces_numeric_columns():
    """pymysql 返回的 None 会让整列变成 object，mean() 会静默失效。

    必须显式转数值 —— 否则「不适用不参与均值」这条保证会被悄悄破坏。
    """
    frame = to_frame([_row(rec_f1=0.5), _row(rec_f1=None)])

    assert frame["rec_f1"].dtype.kind == "f"
    assert frame["rec_f1"].isna().sum() == 1


def test_to_frame_labels_scenarios():
    frame = to_frame([_row(scenario="report"), _row(scenario="seat")])

    assert list(frame["scenario_label"]) == ["推荐报告", "座位查询"]


def test_to_frame_tolerates_missing_detail():
    row = _row()
    row.pop("detail")

    frame = to_frame([row])

    assert frame["detail"].iloc[0] == {}


# ----------------------------------------------------------------------
# KPI
# ----------------------------------------------------------------------

def test_kpi_on_empty_frame_returns_nones_not_zeros():
    kpi = kpi_summary(pd.DataFrame())

    assert kpi["total"] == 0
    assert kpi["pass_rate"] is None
    assert kpi["rec_f1"] is None


def test_kpi_pass_rate_accumulates_across_turns():
    """分子分母跨轮累加，而不是对各轮通过率取平均。

    否则「只判定 2 项就全过」的轮次会和「判定 11 项」的轮次等权重。
    """
    frame = to_frame([_row(passed=2, determined=2), _row(passed=0, determined=10)])

    kpi = kpi_summary(frame)

    assert kpi["pass_rate"] == pytest.approx(2 / 12)
    assert kpi["pass_rate"] != pytest.approx((1.0 + 0.0) / 2)


def test_kpi_means_skip_not_applicable():
    """不适用（NULL）不参与均值 —— 否则知识类问题会把推荐指标一路拉低。"""
    frame = to_frame([
        _row(rec_f1=1.0), _row(rec_f1=None), _row(rec_f1=None), _row(rec_f1=0.0),
    ])

    kpi = kpi_summary(frame)

    assert kpi["rec_f1"] == pytest.approx(0.5)     # 只算有值的两轮
    assert kpi["rec_f1_n"] == 2


def test_kpi_counts_crashed_turns():
    frame = to_frame([_row(), _row(scenario="failed"), _row(scenario="failed")])

    kpi = kpi_summary(frame)

    assert kpi["total"] == 3
    assert kpi["failed"] == 2
    assert kpi["evaluable"] == 1
    assert kpi["failed_rate"] == pytest.approx(2 / 3)


def test_kpi_excludes_crashed_turns_from_pass_rate():
    """崩溃轮不参与通过率 —— 它连回答都没有，拿它当通过率的分母毫无意义。"""
    frame = to_frame([_row(passed=4, determined=4), _row(scenario="failed",
                                                         passed=0, determined=1)])

    assert kpi_summary(frame)["pass_rate"] == 1.0


# ----------------------------------------------------------------------
# 检查项失败分布
# ----------------------------------------------------------------------

def test_check_failure_rates_excludes_not_applicable_from_denominator():
    frame = to_frame([
        _row(checks=_checks(("回答非空", "pass"), ("座位数据来自工具", "na"))),
        _row(checks=_checks(("回答非空", "fail"), ("座位数据来自工具", "na"))),
    ])

    rates = check_failure_rates(frame)

    answer = rates[rates["check_name"] == "回答非空"].iloc[0]
    assert answer["determined"] == 2
    assert answer["failure_rate"] == pytest.approx(0.5)
    assert "座位数据来自工具" not in set(rates["check_name"])


def test_check_failure_rates_sorted_descending():
    frame = to_frame([
        _row(checks=_checks(("A", "fail"), ("B", "pass"), ("C", "fail"))),
        _row(checks=_checks(("A", "pass"), ("B", "pass"), ("C", "fail"))),
    ])

    rates = check_failure_rates(frame)

    assert list(rates["check_name"]) == ["C", "A", "B"]
    assert rates.iloc[0]["failure_rate"] == 1.0


def test_check_failure_rates_on_empty():
    assert check_failure_rates(pd.DataFrame()).empty


# ----------------------------------------------------------------------
# 小样本保护
# ----------------------------------------------------------------------

def test_by_model_separates_small_samples():
    """跑 3 轮的模型不该和跑 20 轮的模型并列排名 —— 那比的是运气。"""
    rows = [_row(model="big")] * 20 + [_row(model="tiny")] * 3

    enough, scarce = by_model(to_frame(rows))

    assert list(enough["model_id"]) == ["big"]
    assert list(scarce["model_id"]) == ["tiny"]
    assert scarce.iloc[0]["n"] == 3


def test_by_model_threshold_boundary():
    """恰好达到阈值就参与对比（阈值是「>=」而不是「>」）。"""
    rows = [_row(model="exactly")] * MIN_SAMPLES_FOR_RANKING

    enough, scarce = by_model(to_frame(rows))

    assert len(enough) == 1
    assert scarce.empty


def test_by_model_excludes_crashed_turns():
    rows = [_row(model="m")] * 2 + [_row(model="m", scenario="failed")] * 10
    enough, scarce = by_model(to_frame(rows))

    assert enough.empty
    assert scarce.iloc[0]["n"] == 2       # 崩溃轮不计入样本量


# ----------------------------------------------------------------------
# 趋势
# ----------------------------------------------------------------------

def test_daily_trend_groups_by_day():
    rows = [_row(day=0), _row(day=0), _row(day=1)]

    trend = daily_trend(to_frame(rows))

    assert len(trend) == 3 - 1
    assert trend["n"].sum() == 3


def test_daily_trend_carries_sample_size():
    """通过率必须和样本量一起给出 —— 1 轮全过和 40 轮全过都是 100%。"""
    # 3/4 与 0/4 合并 → 3/8
    trend = daily_trend(to_frame([_row(day=0), _row(day=0, passed=0, determined=4)]))

    assert "n" in trend.columns
    assert trend["n"].iloc[0] == 2
    assert trend["pass_rate"].iloc[0] == pytest.approx(3 / 8)


def test_daily_trend_on_empty():
    assert daily_trend(pd.DataFrame()).empty


# ----------------------------------------------------------------------
# 场景 × 检查项矩阵
# ----------------------------------------------------------------------

def test_scenario_check_matrix():
    frame = to_frame([
        _row(scenario="report", checks=_checks(("报告流程完整", "pass"))),
        _row(scenario="report", checks=_checks(("报告流程完整", "fail"))),
        _row(scenario="seat", checks=_checks(("报告流程完整", "na"))),
    ])

    matrix = scenario_check_matrix(frame)

    report = matrix[matrix["scenario_label"] == "推荐报告"].iloc[0]
    assert report["pass_rate"] == pytest.approx(0.5)
    assert report["n"] == 2
    assert "座位查询" not in set(matrix["scenario_label"])    # na 不进矩阵


# ----------------------------------------------------------------------
# 规则健康度
# ----------------------------------------------------------------------

def test_other_share_measures_rule_coverage():
    """这是用来审计关键词规则本身的数，不是给模型打的分。"""
    rows = [_row(scenario="other", source="unknown")] * 2 + [_row(scenario="report")] * 2

    share = other_share(to_frame(rows))

    assert share["other_share"] == pytest.approx(0.5)
    assert share["unknown_share"] == pytest.approx(0.5)
    assert share["total"] == 4


def test_top_unsatisfied_groups():
    rows = [_row(unsatisfied=[["fetch_external_data"]])] * 3 + \
           [_row(unsatisfied=[["search_books_by_category", "search_book"]])]

    top = top_unsatisfied_groups(to_frame(rows))

    assert top.iloc[0]["groups"] == "fetch_external_data"
    assert top.iloc[0]["n"] == 3
