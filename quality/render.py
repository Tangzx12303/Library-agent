"""质检结果的展示格式化 —— 纯函数，不含 Streamlit。

放在这里而不是 ``app.py`` 里，是为了能被 pytest 直接测：``app.py`` 一旦被导入
就会执行 ``st.set_page_config`` 和整段视图分发，根本没法在测试里 import。

## 一条贯穿的规则：「不适用」显示成「—」，而不是 0%

读者问「图书馆有哪些功能分区」时，推荐准确率是**不适用**。若显示「0%」，读者会
以为系统认为自己答错了。所以 ``rate(None)`` 返回破折号，并在面板下方用一行
caption 说明它的含义。
"""
from quality.types import STATE_FAIL, STATE_NA, STATE_PASS

# 「不适用」的占位符
NOT_APPLICABLE = "—"

# 状态标签：图标 + 文字。**不能只靠颜色或图标传达状态** —— 色觉障碍用户读不出
# 红绿差异，而图标在没有文字对照时也有歧义。
STATE_LABELS = {
    STATE_PASS: "✅ 通过",
    STATE_FAIL: "❌ 未通过",
    STATE_NA: "➖ 不适用",
}

STATE_ORDER = {STATE_FAIL: 0, STATE_PASS: 1, STATE_NA: 2}


def rate(value: float | None, digits: int = 0) -> str:
    """把比率格式化成百分比；None（不适用）显示成破折号。"""
    if value is None:
        return NOT_APPLICABLE
    return f"{value * 100:.{digits}f}%"


def state_label(state: str) -> str:
    return STATE_LABELS.get(state, state)


def pass_rate_text(passed: int, determined: int) -> str:
    """通过率按「通过/判定」显示。

    分母是**判定过的项数**，不是检查项总数 —— 一轮普通问答只判定四五项，
    剩下的是不适用。写成 ``4/11`` 会让读者误以为漏了七项。
    """
    if not determined:
        return NOT_APPLICABLE
    return f"{passed}/{determined}"


def checks_rows(compact: dict) -> list[dict]:
    """检查项明细表：未通过的排在最前，读者一眼看到问题。

    ``note`` 缺省用空串而非 None —— ``st.dataframe`` 底层的 Arrow 不接受
    一列里既有 str 又有 None（项目已经栽过一次 ArrowTypeError）。
    """
    checks = (compact or {}).get("checks") or []
    rows = [
        {
            "检查项": c.get("name", ""),
            "结果": state_label(c.get("state", STATE_NA)),
            "说明": c.get("note") or "",
            "_order": STATE_ORDER.get(c.get("state", STATE_NA), 9),
        }
        for c in checks
    ]
    rows.sort(key=lambda row: row["_order"])
    for row in rows:
        row.pop("_order", None)
    return rows


def metric_cards(compact: dict) -> list[tuple[str, str, str]]:
    """面板顶部的三个指标卡：``(标题, 值, 说明)``。

    第三项「工具召回」的说明在两处体现口径：它只衡量提示词明令要求的必需工具。
    """
    compact = compact or {}

    if compact.get("error"):
        return [
            ("检查项通过", pass_rate_text(compact.get("passed", 0),
                                       compact.get("determined", 0)),
             "本轮对话未正常完成"),
            ("推荐 F1", NOT_APPLICABLE, "本轮未产生推荐结果"),
            ("工具召回", NOT_APPLICABLE, "本轮未产生工具调用"),
        ]

    return [
        ("检查项通过",
         pass_rate_text(compact.get("passed", 0), compact.get("determined", 0)),
         "仅统计本轮适用的检查项"),
        ("推荐 F1",
         rate(compact.get("rec_f1")),
         "推荐书目与工具给出的在架可借书目的一致性"),
        ("工具召回",
         rate(compact.get("tool_recall")),
         "提示词要求的必需工具是否调用到位"),
    ]


def headline(compact: dict) -> str:
    """折叠面板标题栏上的一句话摘要。"""
    compact = compact or {}

    if compact.get("error"):
        return "本轮对话异常中断"

    passed = compact.get("passed", 0)
    determined = compact.get("determined", 0)
    parts = [f"{pass_rate_text(passed, determined)} 项通过"]

    rec_f1 = compact.get("rec_f1")
    tool_recall = compact.get("tool_recall")
    if rec_f1 is not None:
        parts.append(f"推荐 F1 {rate(rec_f1)}")
    if tool_recall is not None:
        parts.append(f"工具召回 {rate(tool_recall)}")

    return " · ".join(parts)


def caption(compact: dict) -> str:
    """面板底部对「不适用」的一行说明。"""
    compact = compact or {}
    na_count = sum(1 for c in (compact.get("checks") or [])
                   if c.get("state") == STATE_NA)

    base = "「—」表示本轮不适用，不计入任何统计。"
    if na_count:
        return f"{base}本轮有 {na_count} 项检查不适用于该场景。"
    return base


def tools_text(compact: dict) -> str:
    """本轮的工具调用轨迹，按调用次序。"""
    tools = (compact or {}).get("tools_called") or []
    return " → ".join(tools) if tools else "（本轮未调用工具）"


def scope_note() -> str:
    """口径说明 —— 展示数字的地方就该说清数字管什么。

    不写这句，读者（和面试官）会以为「工具召回」是在给模型的整体表现打分。
    """
    return (
        "指标口径：推荐准确率 = 推荐书目中来自工具返回的在架可借书目的比例"
        "（理想值 100%）；推荐召回率 = 该类别在架可借书目被推荐出的比例；"
        "工具召回率只衡量提示词明令要求的必需工具是否调用到位，"
        "不惩罚模型额外调用的合理解释性工具。"
    )
