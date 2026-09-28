"""质检结果的数据结构。

``passed`` 用**三态**而不是布尔：``True`` 通过 / ``False`` 未通过 / ``None``
不适用。「本轮没有图书推荐」时的「推荐书目均标注索书号」是**不适用**，
把它算成通过会虚高通过率，算成未通过会冤枉模型——所以必须能表达第三种状态。
"""
import json
from dataclasses import asdict, dataclass, field

from quality.scenario import Scenario

# 通过率的分母只算「判定过」的项，不含不适用
STATE_PASS = "pass"
STATE_FAIL = "fail"
STATE_NA = "na"


@dataclass
class CheckResult:
    name: str
    passed: bool | None
    note: str = ""

    @property
    def state(self) -> str:
        if self.passed is None:
            return STATE_NA
        return STATE_PASS if self.passed else STATE_FAIL

    @property
    def determined(self) -> bool:
        return self.passed is not None


@dataclass
class EvaluationReport:
    """一轮对话的质检报告。"""

    query: str
    scenario: Scenario
    checks: list[CheckResult] = field(default_factory=list)

    # 推荐质量（无推荐时全部为 None）
    rec_precision: float | None = None
    rec_recall: float | None = None
    rec_f1: float | None = None

    # 工具使用（无必需工具时 tool_recall 为 None）
    tool_precision: float | None = None
    tool_recall: float | None = None
    tool_f1: float | None = None
    extra_call_rate: float | None = None

    # 归因用的明细
    tools_called: list[str] = field(default_factory=list)
    unsatisfied_groups: list[list[str]] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    recommended: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)

    # 元信息
    model_id: str = ""
    reader_id: str = ""
    latency_ms: int | None = None
    error: str | None = None

    # ----------------------------------------------------------------
    # 通过率
    # ----------------------------------------------------------------
    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def determined_count(self) -> int:
        """判定过的检查项数 = 分母。

        分母**不是**检查项总数：一轮普通问答只会判定 5–8 项，剩下的不适用。
        拿固定总数当分母会让通过率随场景漂移，横向比较就失去意义。
        """
        return sum(1 for c in self.checks if c.determined)

    @property
    def not_applicable_count(self) -> int:
        return sum(1 for c in self.checks if not c.determined)

    @property
    def pass_rate(self) -> float | None:
        if not self.determined_count:
            return None
        return self.passed_count / self.determined_count

    @property
    def is_failed_turn(self) -> bool:
        return self.error is not None

    # ----------------------------------------------------------------
    # 持久化 / 展示
    # ----------------------------------------------------------------
    def to_row(self) -> dict:
        """转成 ``eval_turn_records`` 的一行。

        **只存派生事实，绝不存工具返回的原始文本** —— ``fetch_external_data``
        的 JSON 加 RAG 摘要每轮几 KB，落库会让表涨得很快，而质检并不需要它们。
        """
        return {
            "reader_id": self.reader_id,
            "model_id": self.model_id,
            # 超长问题会在 INSERT 阶段失败，这里先截断
            "query": (self.query or "")[:512],
            "scenario": self.scenario.key,
            "scenario_source": self.scenario.source,
            "passed": self.passed_count,
            "determined": self.determined_count,
            "rec_precision": self.rec_precision,
            "rec_recall": self.rec_recall,
            "rec_f1": self.rec_f1,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "tool_f1": self.tool_f1,
            "latency_ms": self.latency_ms,
            "detail_json": json.dumps(self.to_detail(), ensure_ascii=False),
        }

    def to_detail(self) -> dict:
        return {
            "checks": [
                {"name": c.name, "state": c.state, "note": c.note} for c in self.checks
            ],
            "tools_called": self.tools_called,
            "unsatisfied_groups": self.unsatisfied_groups,
            "candidates": self.candidates,
            "recommended": self.recommended,
            "hallucinated": self.hallucinated,
            "negated": self.negated,
            "extra_call_rate": self.extra_call_rate,
            "matched_keyword": self.scenario.matched,
            "error": self.error,
        }

    def to_compact(self) -> dict:
        """给 ``session_state`` 用的精简版：能在多轮对话里逐条重放而不占内存。

        不含候选书目等长列表 —— 那些只在落库的 detail_json 里保留。
        """
        return {
            "scenario_key": self.scenario.key,
            "scenario_label": self.scenario.label,
            "passed": self.passed_count,
            "determined": self.determined_count,
            "rec_precision": self.rec_precision,
            "rec_recall": self.rec_recall,
            "rec_f1": self.rec_f1,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "tool_f1": self.tool_f1,
            "checks": [
                {"name": c.name, "state": c.state, "note": c.note} for c in self.checks
            ],
            "tools_called": self.tools_called,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    def as_dict(self) -> dict:
        return asdict(self)
