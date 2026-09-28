"""把一次对话的追踪数据，变成一份质检报告。

这是纯计算：不连数据库、不碰 Streamlit，因此可以被 pytest 用假数据直接测。
落库与展示分别由 ``quality.store`` 和 ``quality.render`` 负责。
"""
from quality import metrics
from quality.checks import CheckContext, run_checks
from quality.recommendation import analyze_recommendations
from quality.scenario import Scenario, detect_scenario
from quality.trace import called_names, sorted_calls
from quality.types import EvaluationReport


def evaluate_turn(query: str,
                  response: str,
                  trace: list[dict] | None = None,
                  reader_id: str | None = None,
                  model_id: str = "",
                  latency_ms: int | None = None) -> EvaluationReport:
    """对一轮对话做质检。

    :param query: 读者本轮提问
    :param response: 模型的完整回答文本（``st.write_stream`` 的返回值）
    :param trace: 本轮的工具调用轨迹（由中间件在流式过程中填充）
    :param reader_id: 当前登录读者，用于核对借阅查询有没有越权
    :param latency_ms: 本轮总耗时
    """
    trace = sorted_calls(trace)
    scenario: Scenario = detect_scenario(query, trace)
    recommendation = analyze_recommendations(response, trace)

    ctx = CheckContext(query=query, response=response, trace=trace,
                       scenario=scenario, recommendation=recommendation,
                       reader_id=reader_id)
    checks = run_checks(ctx)

    report = EvaluationReport(
        query=query, scenario=scenario, checks=checks,
        tools_called=called_names(trace),
        candidates=recommendation.candidates,
        recommended=recommendation.recommended,
        hallucinated=recommendation.hallucinated,
        negated=recommendation.negated,
        model_id=model_id, reader_id=reader_id or "",
        latency_ms=latency_ms,
    )

    _fill_recommendation_metrics(report, recommendation)
    _fill_tool_metrics(report, scenario)
    return report


def _fill_recommendation_metrics(report: EvaluationReport, recommendation) -> None:
    """推荐质量：准确率看「有没有编造」，召回率看「候选覆盖得全不全」。

    两者都以**候选集**（工具给过的在架可借书目）为真值。没有推荐或没有候选时
    全部留 None —— 落库后是 NULL，MySQL 的 AVG() 会自动跳过，不会污染均值。
    """
    recommended = set(recommendation.recommended)
    candidates = set(recommendation.candidates)

    report.rec_precision = metrics.precision(recommended, candidates)
    report.rec_recall = metrics.recall(recommended, candidates)
    report.rec_f1 = metrics.f1_score(report.rec_precision, report.rec_recall)


def _fill_tool_metrics(report: EvaluationReport, scenario: Scenario) -> None:
    """工具使用：召回率按 OR 组算，合规率按禁用集算。

    刻意**不用**「期望工具集」算准确率 —— 那会把模型多做一件合理的事判成失误。
    详细理由见 ``quality/metrics.py`` 的 ``compliance_rate``。
    """
    actual = {name for name in report.tools_called if name}

    tool_recall, unsatisfied = metrics.group_recall(actual, scenario.required)
    report.tool_recall = tool_recall
    report.unsatisfied_groups = [list(group) for group in unsatisfied]

    report.tool_precision = metrics.compliance_rate(actual, set(scenario.forbidden))
    report.tool_f1 = metrics.f1_score(report.tool_precision, report.tool_recall)
    report.extra_call_rate = metrics.extra_call_rate(actual, scenario.required_flat)
