"""把质检记录聚合成图表数据 —— 纯 pandas，不含 Streamlit。

放在这里而不是 ``app.py`` 里，是为了能被 pytest 直接测：``app.py`` 一被导入就会
执行 ``st.set_page_config`` 和整段视图分发。

## 几条刻意的取舍

- **均值只用有效样本**：``rec_f1`` 为 NULL 的轮次不参与推荐 F1 均值（pandas 的
  ``mean()`` 默认跳过 NaN，与 MySQL 的 ``AVG()`` 行为一致）。
- **小样本不排名**：某模型只跑过 3 轮就不能和跑了 200 轮的模型并列比高低 —— 那
  比的是运气。样本不足的模型单独列出来，不进对比图。
- **不把「不适用」当 0**：整列全为 NaN 时返回 NaN，由展示层渲染成「—」。
"""
import pandas as pd

from quality.scenario import FAILED, OTHER, SOURCE_UNKNOWN, spec_of

# 模型参与横向对比所需的最少轮次
MIN_SAMPLES_FOR_RANKING = 5

# 参与聚合的列
_METRIC_COLUMNS = ("rec_precision", "rec_recall", "rec_f1",
                   "tool_precision", "tool_recall", "tool_f1")


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """把 DAO 读出来的行规整成 DataFrame，并展开每轮的检查项明细。"""
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)

    frame["created_at"] = pd.to_datetime(frame.get("created_at"), errors="coerce")
    frame["date"] = frame["created_at"].dt.date

    # 比率列显式转数值：pymysql 给回的可能是 None，混在 object 列里会让 mean() 失效
    for column in _METRIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = float("nan")

    frame["passed"] = pd.to_numeric(frame.get("passed"), errors="coerce").fillna(0)
    frame["determined"] = pd.to_numeric(frame.get("determined"), errors="coerce").fillna(0)

    detail = frame.get("detail")
    if detail is None:
        detail = pd.Series([{}] * len(frame), index=frame.index)
    frame["detail"] = detail.apply(lambda value: value if isinstance(value, dict) else {})

    frame["scenario_label"] = frame["scenario"].map(lambda key: spec_of(key).label)
    return frame


def kpi_summary(frame: pd.DataFrame) -> dict:
    """看板顶部那排数字。

    ``pass_rate`` 的分子分母是**跨轮累加**的 ``sum(passed)/sum(determined)``，
    而不是各轮通过率的平均 —— 后者会让只判定 2 项就全过的轮次和判定 11 项的轮次
    等权重。
    """
    if frame.empty:
        return {
            "total": 0, "failed": 0, "failed_rate": None, "evaluable": 0,
            "pass_rate": None, "rec_f1": None, "tool_recall": None,
            "rec_f1_n": 0, "tool_recall_n": 0, "unknown_share": None,
        }

    failed_mask = frame["scenario"] == FAILED
    ok = frame[~failed_mask]

    determined = ok["determined"].sum()
    passed = ok["passed"].sum()

    rec = ok["rec_f1"].dropna()
    tool = ok["tool_recall"].dropna()

    return {
        "total": int(len(frame)),
        "failed": int(failed_mask.sum()),
        "failed_rate": _safe_ratio(int(failed_mask.sum()), int(len(frame))),
        "evaluable": int(len(ok)),
        "pass_rate": _safe_ratio(int(passed), int(determined)),
        # 只对**有值**的轮次求均值；count 一并给出，caption 里要写明样本量
        "rec_f1": float(rec.mean()) if len(rec) else None,
        "rec_f1_n": int(len(rec)),
        "tool_recall": float(tool.mean()) if len(tool) else None,
        "tool_recall_n": int(len(tool)),
        "unknown_share": _safe_ratio(
            int((frame["scenario_source"] == SOURCE_UNKNOWN).sum()), int(len(frame))),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def check_failure_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """每个检查项的失败率，降序 —— 全页信息量最高的一张图。

    它直接指出该改提示词的哪一句：失败率整体偏高且平坦 → 提示词的问题；
    只有某一项突出 → 那条规则的问题。
    """
    if frame.empty:
        return pd.DataFrame(columns=["check_name", "determined", "failed", "failure_rate"])

    records: list[dict] = []
    for _, row in frame[~frame["scenario"].isin([FAILED])].iterrows():
        for check in row["detail"].get("checks") or []:
            state = check.get("state")
            if state not in ("pass", "fail"):
                continue          # 不适用不计入分母
            records.append({"check_name": check.get("name", "?"), "failed": state == "fail"})

    if not records:
        return pd.DataFrame(columns=["check_name", "determined", "failed", "failure_rate"])

    grouped = (pd.DataFrame(records)
               .groupby("check_name")
               .agg(determined=("failed", "size"), failed=("failed", "sum"))
               .reset_index())
    grouped["failure_rate"] = grouped["failed"] / grouped["determined"]
    return grouped.sort_values(["failure_rate", "check_name"], ascending=[False, True],
                               ignore_index=True)


def by_model(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按模型聚合，返回 ``(可对比的, 样本不足的)``。

    样本不足的模型**不参与横向排名**并把 n 一并带出 —— 跑了 3 轮的模型不该在图上
    压过跑了 200 轮的模型，那比的是运气而不是质量。
    """
    columns = ["model_id", "n", "pass_rate", "tool_recall", "rec_f1"]
    if frame.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    ok = frame[~frame["scenario"].isin([FAILED])]
    grouped = (ok.groupby("model_id")
               .agg(n=("passed", "size"),
                    passed=("passed", "sum"),
                    determined=("determined", "sum"),
                    tool_recall=("tool_recall", "mean"),
                    rec_f1=("rec_f1", "mean"))
               .reset_index())
    grouped["pass_rate"] = grouped["passed"] / grouped["determined"].where(
        grouped["determined"] > 0)
    grouped = grouped[columns]

    enough = grouped[grouped["n"] >= MIN_SAMPLES_FOR_RANKING].sort_values(
        "tool_recall", ascending=False, ignore_index=True)
    scarce = grouped[grouped["n"] < MIN_SAMPLES_FOR_RANKING].sort_values(
        "n", ascending=False, ignore_index=True)
    return enough, scarce


def daily_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """按天聚合出通过率与样本量。

    样本量必须和通过率一起画：某天只有 1 轮且全过 → 100%，和 40 轮全过的 100%
    完全是两回事。所以趋势拆成上下两张共享 x 轴的图，**不做双 y 轴**。
    """
    columns = ["date", "n", "pass_rate"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    ok = frame[~frame["scenario"].isin([FAILED])]
    if ok.empty:
        return pd.DataFrame(columns=columns)

    grouped = (ok.groupby("date")
               .agg(n=("passed", "size"),
                    passed=("passed", "sum"),
                    determined=("determined", "sum"))
               .reset_index())
    grouped["pass_rate"] = grouped["passed"] / grouped["determined"].where(
        grouped["determined"] > 0)
    return grouped[columns].sort_values("date", ignore_index=True)


def scenario_check_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """场景 × 检查项的通过率矩阵。

    这是**发现规则错误**的唯一手段：如果某个场景下所有轮次都在同一项上失败，
    该先怀疑「这个场景的必需工具组定义错了」，而不是模型不行。
    """
    if frame.empty:
        return pd.DataFrame(columns=["scenario_label", "check_name", "pass_rate", "n"])

    records: list[dict] = []
    for _, row in frame[~frame["scenario"].isin([FAILED])].iterrows():
        for check in row["detail"].get("checks") or []:
            state = check.get("state")
            if state not in ("pass", "fail"):
                continue
            records.append({
                "scenario_label": row["scenario_label"],
                "check_name": check.get("name", "?"),
                "passed": state == "pass",
            })

    if not records:
        return pd.DataFrame(columns=["scenario_label", "check_name", "pass_rate", "n"])

    grouped = (pd.DataFrame(records)
               .groupby(["scenario_label", "check_name"])
               .agg(n=("passed", "size"), pass_rate=("passed", "mean"))
               .reset_index())
    return grouped


def top_unsatisfied_groups(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """最常没被满足的必需工具组 —— 直接指向「模型漏调了哪个工具」。"""
    if frame.empty:
        return pd.DataFrame(columns=["groups", "n"])

    counter: dict[str, int] = {}
    for _, row in frame.iterrows():
        for group in row["detail"].get("unsatisfied_groups") or []:
            key = " 或 ".join(group) if isinstance(group, list) else str(group)
            counter[key] = counter.get(key, 0) + 1

    if not counter:
        return pd.DataFrame(columns=["groups", "n"])
    return (pd.DataFrame(counter.items(), columns=["groups", "n"])
            .sort_values("n", ascending=False, ignore_index=True).head(limit))


def other_share(frame: pd.DataFrame) -> dict:
    """场景判定的健康度。

    ``other`` 占比高说明关键词表覆盖不足；这些轮次的指标是「不适用」，
    不会造成错误扣分，但覆盖不到就发现不了问题 —— 所以这个数要显示出来，
    它是**用来审计规则本身**的，不是用来给模型打分的。
    """
    if frame.empty:
        return {"other_share": None, "unknown_share": None, "total": 0}

    total = len(frame)
    return {
        "other_share": _safe_ratio(int((frame["scenario"] == OTHER).sum()), total),
        "unknown_share": _safe_ratio(
            int((frame["scenario_source"] == SOURCE_UNKNOWN).sum()), total),
        "total": total,
    }
