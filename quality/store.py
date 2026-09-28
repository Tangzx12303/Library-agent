"""质检的持久化门面 —— **唯一会写数据库的地方，且永不抛异常**。

质检是**旁路**功能：它失败绝不能让读者看不到回答。所以这里的每个入口都用
try/except 包住，失败只记日志。

顺带守住一个容易被忽略的统计偏差：**崩溃轮次也必须留一行记录**。工具抛异常时
LangGraph 会中断整轮执行，``response`` 为 ``None``，如果没有 ``safe_record_failure``，
那一轮就会从表里彻底消失——结果是**数据库越坏，管理员的看板看起来越好**。
"""
import json

from db import repository as repo
from quality.evaluator import evaluate_turn
from quality.scenario import failed_scenario
from quality.types import CheckResult, EvaluationReport
from utils.logger_handler import logger

# 崩溃轮的检查项名称
CRASH_CHECK_NAME = "本轮对话未因异常中断"


def safe_evaluate(query: str,
                  response: str,
                  trace: list[dict] | None = None,
                  reader_id: str | None = None,
                  model_id: str = "",
                  latency_ms: int | None = None) -> EvaluationReport | None:
    """质检 + 落库，全程不抛异常。

    返回报告供界面展示；任何一步失败返回 None（读者侧静默跳过质检面板）。
    **落库失败也要返回报告**——数据能不能存下来，不该影响读者看不看得到结果。
    """
    try:
        report = evaluate_turn(query=query, response=response, trace=trace,
                              reader_id=reader_id, model_id=model_id,
                              latency_ms=latency_ms)
    except Exception as exc:                          # noqa: BLE001 —— 兜底是刻意的
        logger.error(f"[质检]指标计算失败，本轮跳过：{type(exc).__name__}: {exc}",
                     exc_info=True)
        return None

    try:
        repo.insert_eval_turn_record(report.to_row())
    except Exception as exc:                          # noqa: BLE001
        logger.error(f"[质检]记录落库失败（对话不受影响）：{type(exc).__name__}: {exc}")

    return report


def safe_record_failure(query: str,
                        error: str,
                        reader_id: str | None = None,
                        model_id: str = "",
                        latency_ms: int | None = None) -> None:
    """为异常中断的轮次补一条记录。

    这一行不参与任何比率（``scenario='failed'``，所有指标列为 NULL），只用于
    统计**崩溃率**，并让通过率的分母诚实——否则崩溃的轮次会凭空消失，
    看板看上去会随着系统变坏而变好。
    """
    report = EvaluationReport(
        query=query or "",
        scenario=failed_scenario(),
        checks=[CheckResult(CRASH_CHECK_NAME, False, error or "未知异常")],
        model_id=model_id or "",
        reader_id=reader_id or "",
        latency_ms=latency_ms,
        error=error or "未知异常",
    )

    try:
        repo.insert_eval_turn_record(report.to_row())
    except Exception as exc:                          # noqa: BLE001
        logger.error(f"[质检]崩溃轮次记录落库失败：{type(exc).__name__}: {exc}")


def safe_load_records(**kwargs) -> list[dict]:
    """读取质检记录，失败返回空列表 —— 看板宁可空白也不该整页崩。

    读出来的 ``detail_json`` 是字符串，这里顺手解析成 dict，并补一个可读的
    场景标签，省得每个调用点各写一遍。
    """
    try:
        rows = repo.list_eval_records(**kwargs)
    except Exception as exc:                          # noqa: BLE001
        logger.error(f"[质检]读取记录失败（看板将显示为空）：{type(exc).__name__}: {exc}")
        return []

    for row in rows:
        try:
            row["detail"] = json.loads(row.get("detail_json") or "{}")
        except (TypeError, ValueError):
            row["detail"] = {}
    return rows


def safe_count_records(since: str | None = None) -> int:
    try:
        return repo.count_eval_records(since)
    except Exception as exc:                          # noqa: BLE001
        logger.error(f"[质检]统计记录数失败：{type(exc).__name__}: {exc}")
        return 0


def safe_list_models() -> list[str]:
    try:
        return repo.list_eval_models()
    except Exception as exc:                          # noqa: BLE001
        logger.error(f"[质检]读取模型清单失败：{type(exc).__name__}: {exc}")
        return []
