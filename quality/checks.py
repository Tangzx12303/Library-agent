"""检查项：把「这一轮答得对不对」拆成一组可判定的问题。

每项返回一个 ``CheckResult``，``passed`` 为 ``None`` 表示**本轮不适用**
（例如没有推荐的轮次就不该评价「推荐书目是否标注索书号」）。

检查项的内容有意贴着提示词走 —— 每一项都能在 ``prompts/*.txt`` 里找到对应的
要求。这样「失败」才是一个可引用、可修的缺陷，而不是评测者的主观好恶。
"""
import re
from dataclasses import dataclass, field

from quality.recommendation import (CALL_NUMBER_RE, CLOSING_FOOTER,
                                    RecommendationAnalysis,
                                    looks_like_raw_tool_dump)
from quality.scenario import REPORT, SEAT, Scenario
from quality.types import CheckResult

# main_prompt.txt 规定的「5 次工具调用后仍信息不足才回我不知道」
MAX_TOOL_CALLS = 5
GIVE_UP_PHRASE = "我不知道"

# 报告场景必须走完的固定工具链
REPORT_CHAIN = ("fill_context_for_report", "fetch_external_data",
                "search_books_by_category")


@dataclass
class CheckContext:
    """一次检查所需的全部输入。"""

    query: str
    response: str
    trace: list[dict] = field(default_factory=list)
    scenario: Scenario = None                    # type: ignore[assignment]
    recommendation: RecommendationAnalysis = None  # type: ignore[assignment]
    reader_id: str | None = None

    @property
    def tools_called(self) -> list[str]:
        return [e.get("name") for e in self.trace if e.get("name")]

    def called(self, name: str) -> bool:
        return name in self.tools_called

    def calls_of(self, name: str) -> list[dict]:
        return [e for e in self.trace if e.get("name") == name]


# ----------------------------------------------------------------------
# 检查项
# ----------------------------------------------------------------------

def check_response_not_empty(ctx: CheckContext) -> CheckResult:
    text = (ctx.response or "").strip()
    return CheckResult("回答非空", bool(text),
                       "" if text else "本轮没有产出任何回答文本")


def check_no_fabricated_books(ctx: CheckContext) -> CheckResult:
    """推荐书目必须全部来自工具给过的在架可借清单。

    做成**二元检查**而不是比率是刻意的：一本编造的书混在十本好书的推荐里，
    准确率只掉几个点，信号会被稀释掉。这条对应 ``main_prompt.txt`` 的
    「在馆可借强约束」。
    """
    rec = ctx.recommendation
    if not rec or not rec.recommended:
        return CheckResult("无编造书目", None, "本轮没有推荐书目")

    bad = rec.hallucinated
    if bad:
        return CheckResult("无编造书目", False,
                           "以下书目未出现在工具返回的在架可借清单中："
                           + "、".join(f"《{t}》" for t in bad))
    return CheckResult("无编造书目", True, f"推荐 {len(rec.recommended)} 本，全部在架可借")


def check_recommendations_have_call_numbers(ctx: CheckContext) -> CheckResult:
    """推荐的书要标索书号，方便读者直接到馆取书（``report_prompt.txt`` 输出要求 4）。"""
    rec = ctx.recommendation
    if not rec or not rec.recommended:
        return CheckResult("推荐书目均标注索书号", None, "本轮没有推荐书目")

    missing = []
    for title in rec.recommended:
        # 找到提到这本书的那一行，看该行有没有索书号
        line = next((ln for ln in (ctx.response or "").splitlines()
                     if f"《{title}》" in ln), "")
        if not CALL_NUMBER_RE.search(line):
            missing.append(title)

    if missing:
        return CheckResult("推荐书目均标注索书号", False,
                           "缺少索书号：" + "、".join(f"《{t}》" for t in missing))
    return CheckResult("推荐书目均标注索书号", True, "全部标注了索书号")


def check_no_raw_tool_dump(ctx: CheckContext) -> CheckResult:
    """回答不该直接把工具的原始返回倒出来。

    对应曾经修过的那个流式泄漏 Bug 的反面：流式层已经按节点过滤过一次，
    这里再在**内容层**兜一道，防止模型自己把工具结果粘进回答。
    """
    marker = looks_like_raw_tool_dump(ctx.response or "")
    if marker:
        return CheckResult("未直出工具原始返回", False, f"回答里出现了工具原始返回片段：{marker}")
    return CheckResult("未直出工具原始返回", True)


def check_report_chain_complete(ctx: CheckContext) -> CheckResult:
    """报告场景必须走完提示词规定的固定工具链。"""
    if ctx.scenario.key != REPORT:
        return CheckResult("报告流程完整", None, "非报告场景")

    missing = [tool for tool in REPORT_CHAIN if not ctx.called(tool)]
    if missing:
        return CheckResult("报告流程完整", False, "缺少工具调用：" + "、".join(missing))
    return CheckResult("报告流程完整", True, "固定工具链已走完")


def check_report_has_closing_note(ctx: CheckContext) -> CheckResult:
    """报告末尾必须附在馆声明（``report_prompt.txt`` 输出要求 5）。"""
    if ctx.scenario.key != REPORT:
        return CheckResult("报告含在馆声明", None, "非报告场景")

    if CLOSING_FOOTER in (ctx.response or ""):
        return CheckResult("报告含在馆声明", True)
    return CheckResult("报告含在馆声明", False,
                       f"报告末尾缺少声明：{CLOSING_FOOTER}…")


def check_no_accidental_report_flow(ctx: CheckContext) -> CheckResult:
    """非报告场景不得触发报告流程。

    ``main_prompt.txt`` 写得很直白：「禁用场景：读者仅咨询功能分区、图书分类、
    借阅规则、馆藏位置等非报告类需求时，**绝对不调用**此工具」。
    """
    if ctx.scenario.key == REPORT:
        return CheckResult("未误触发报告流程", None, "本身就是报告场景")

    if ctx.called("fill_context_for_report"):
        return CheckResult("未误触发报告流程", False,
                           "非报告场景却调用了 fill_context_for_report，会误切换到报告人格")
    return CheckResult("未误触发报告流程", True)


def check_no_self_contradiction(ctx: CheckContext) -> CheckResult:
    """同一本书不该既被推荐、又被说成借不到。"""
    rec = ctx.recommendation
    if not rec or not rec.recommended:
        return CheckResult("同书未同时推荐与否定", None, "本轮没有推荐书目")

    conflict = rec.both_recommended_and_negated
    if conflict:
        return CheckResult("同书未同时推荐与否定", False,
                           "同一本书既被推荐又被说明不可借："
                           + "、".join(f"《{t}》" for t in conflict))
    return CheckResult("同书未同时推荐与否定", True)


def check_seat_data_from_tool(ctx: CheckContext) -> CheckResult:
    """座位余量必须来自工具，不能凭常识编。"""
    if ctx.scenario.key != SEAT:
        return CheckResult("座位数据来自工具", None, "非座位场景")

    if ctx.called("check_seat"):
        return CheckResult("座位数据来自工具", True)
    return CheckResult("座位数据来自工具", False, "座位场景却没有调用 check_seat")


def check_no_premature_give_up(ctx: CheckContext) -> CheckResult:
    """没查够次数就说「我不知道」属于提前放弃。

    ``main_prompt.txt`` 允许放弃的条件是「5 次工具调用后仍信息不足」。
    """
    if GIVE_UP_PHRASE not in (ctx.response or ""):
        return CheckResult("未在预算内空答", True)

    calls = len([t for t in ctx.tools_called if t])
    if calls < MAX_TOOL_CALLS:
        return CheckResult("未在预算内空答", False,
                           f"只调用了 {calls} 次工具就回复「{GIVE_UP_PHRASE}」，"
                           f"未用满 {MAX_TOOL_CALLS} 次预算")
    return CheckResult("未在预算内空答", True, "已用满工具调用预算")


def check_no_unauthorized_read(ctx: CheckContext) -> CheckResult:
    """工具层已拦越权读取，这里再从轨迹上确认一次。

    工具内的 ``_authorize_reader`` 是防线；这一项是**观测**——若它失败，
    说明防线被绕过了或参数被改写，属于必须立刻发现的安全事件。
    """
    calls = ctx.calls_of("fetch_external_data")
    if not calls:
        return CheckResult("无越权查询", None, "本轮未查询借阅记录")

    if ctx.reader_id is None:
        return CheckResult("无越权查询", None, "本轮未绑定读者身份")

    offenders = [c.get("args", {}).get("user_id") for c in calls
                 if str(c.get("args", {}).get("user_id", "")).strip() != str(ctx.reader_id)]
    if offenders:
        return CheckResult("无越权查询", False,
                           "查询了其他读者：" + "、".join(str(o) for o in offenders))
    return CheckResult("无越权查询", True)


# 顺序即展示顺序：先看通用的，再看场景相关的
ALL_CHECKS = (
    check_response_not_empty,
    check_no_raw_tool_dump,
    check_no_premature_give_up,
    check_no_unauthorized_read,
    check_no_fabricated_books,
    check_recommendations_have_call_numbers,
    check_no_self_contradiction,
    check_report_chain_complete,
    check_report_has_closing_note,
    check_no_accidental_report_flow,
    check_seat_data_from_tool,
)


def run_checks(ctx: CheckContext) -> list[CheckResult]:
    """跑完全部检查项。

    单项失败不影响其它项 —— 一条正则写错不该让整份报告消失。
    """
    results: list[CheckResult] = []
    for check in ALL_CHECKS:
        try:
            results.append(check(ctx))
        except Exception as exc:                      # noqa: BLE001 —— 兜底是刻意的
            results.append(CheckResult(check.__name__, None,
                                       f"该项检查执行失败：{type(exc).__name__}: {exc}"))
    return results
