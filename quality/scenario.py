"""场景判定与每个场景的「必需工具组 / 禁用工具」。

**ground truth 来自提示词自身的契约**，不是我的主观猜测：``main_prompt.txt``
要求报告场景必须走固定工具链、非报告场景绝对不调 ``fill_context_for_report``；
``report_prompt.txt`` 要求推荐前必须调 ``search_books_by_category``。检查项就是
在核对这些契约有没有被遵守。

## 关键词表为什么必须窄

场景判错会造成**假扣分**：如果读者问「自习区的开放时间」而规则判成 seat，
模型正确地调了 ``get_opening_hours``，必需组 ``{check_seat}`` 就没人满足，
``tool_recall`` 变成 0 —— 模型答对了却被判失误。

所以这里只收**没有第二种读法**的词，顺序上也做了避让（``开放时间`` 排在
``自习区`` 之前）。任何模糊的提问都会落到 ``other``，而 ``other`` 的必需组为空
→ 指标是"不适用" → **误判永远不会产生错误扣分**，只会少算覆盖。

规则本身是否可靠，靠 ``scenario_source`` 落库后的 ``unknown`` 占比来审计
（见管理员看板的说明），而不是靠信任这张表。
"""
import re
from dataclasses import dataclass, field

# 场景键
REPORT = "report"
CATALOG = "catalog"
SEAT = "seat"
HOURS = "hours"
KNOWLEDGE = "knowledge"
OTHER = "other"
FAILED = "failed"

# 判定来源
SOURCE_TRACE = "trace"
SOURCE_KEYWORD = "keyword"
SOURCE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    label: str
    # 必需工具组：组内 OR（命中任一即算满足），组间 AND（每组都要有人满足）
    required: tuple[tuple[str, ...], ...]
    # 禁用工具：提示词明令本场景不得调用的工具
    forbidden: tuple[str, ...]
    # 关键词（按 SCENARIO_ORDER 的先后依次尝试，先命中者胜）
    keywords: re.Pattern | None = None


@dataclass
class Scenario:
    key: str
    label: str
    source: str
    required: tuple[tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    matched: str = ""          # 命中的关键词（便于排查规则）

    @property
    def required_flat(self) -> set[str]:
        """把 OR 组展开成扁平集合，供「额外调用率」使用。"""
        return {tool for group in self.required for tool in group}


# 「非报告场景禁用报告工具」—— 这条来自 main_prompt.txt 的
# 「禁用场景：读者仅咨询功能分区…时，绝对不调用此工具」。
_NOT_REPORT = ("fill_context_for_report",)


SCENARIOS: dict[str, ScenarioSpec] = {
    REPORT: ScenarioSpec(
        key=REPORT,
        label="推荐报告",
        # 报告流程的固定工具链（main_prompt.txt 第 4 条强约束）。
        # fetch_external_data 与 search_books_by_category 各成一组，
        # 因为提示词明确要求两者都调；最后一个组是 OR，允许用 search_book 核实。
        required=(
            ("fill_context_for_report",),
            ("fetch_external_data",),
            ("search_books_by_category", "search_book"),
        ),
        forbidden=(),
    ),
    CATALOG: ScenarioSpec(
        key=CATALOG,
        label="馆藏查询",
        required=(("search_book", "search_books_by_category"),),
        forbidden=_NOT_REPORT,
    ),
    SEAT: ScenarioSpec(
        key=SEAT,
        label="座位查询",
        required=(("check_seat",),),
        forbidden=_NOT_REPORT,
    ),
    HOURS: ScenarioSpec(
        key=HOURS,
        label="开放时间",
        required=(("get_opening_hours", "rag_summarize"),),
        forbidden=_NOT_REPORT,
    ),
    KNOWLEDGE: ScenarioSpec(
        key=KNOWLEDGE,
        label="知识问答",
        required=(("rag_summarize",),),
        forbidden=_NOT_REPORT,
    ),
    OTHER: ScenarioSpec(
        key=OTHER,
        label="其它",
        required=(),               # 空 → 指标不适用 → 误判不产生扣分
        forbidden=_NOT_REPORT,
    ),
    FAILED: ScenarioSpec(
        key=FAILED,
        label="异常中断",
        required=(),
        forbidden=(),
    ),
}

# 按此顺序依次尝试匹配；先命中者胜出。
# 顺序里做了避让：「自习区的开放时间」应判 hours 而非 seat。
MATCH_ORDER = (REPORT, HOURS, SEAT, CATALOG, KNOWLEDGE)

_PATTERNS: dict[str, re.Pattern] = {
    # 刻意**不收**裸的「推荐」二字：「再推荐几本科幻类的」是报告对话的追问，
    # 它不该被要求重走一遍 fill_context_for_report 的完整流程。
    REPORT: re.compile(
        r"借阅数据|借阅记录|借阅历史|个性化推荐|推荐报告|月度报告|阅读报告|借阅报告"),
    HOURS: re.compile(r"开放时间|开馆|闭馆|几点开门|营业时间|开放到"),
    SEAT: re.compile(r"座位|自习区|阅览区"),
    CATALOG: re.compile(r"《|馆藏|索书号|可以借|能借|借得到|在馆"),
    KNOWLEDGE: re.compile(
        r"功能分区|借阅规则|办证|规章制度|逾期|借书证|续借|图书馆.*规定"),
}


def detect_scenario(query: str, trace: list[dict] | None = None) -> Scenario:
    """判定本轮场景。

    :param query: 读者本轮提问
    :param trace: 工具调用轨迹。``fill_context_for_report`` 出现在轨迹里是
                  **模型自己声明的事实**，比关键词推断可靠；但它只作为佐证，
                  不覆盖关键词判定 —— 若模型在非报告场景误调了它，我们要让
                  「未误触发报告流程」这条检查报警，而不是把场景改判成报告
                  把问题糊过去。
    """
    text = query or ""

    for key in MATCH_ORDER:
        pattern = _PATTERNS.get(key)
        if pattern is None:
            continue
        match = pattern.search(text)
        if match:
            spec = SCENARIOS[key]
            return Scenario(key=spec.key, label=spec.label, source=SOURCE_KEYWORD,
                            required=spec.required, forbidden=spec.forbidden,
                            matched=match.group(0))

    spec = SCENARIOS[OTHER]
    return Scenario(key=spec.key, label=spec.label, source=SOURCE_UNKNOWN,
                    required=spec.required, forbidden=spec.forbidden)


def failed_scenario() -> Scenario:
    """异常中断轮次（对话没跑完），单独一个场景，不参与任何比率统计。"""
    spec = SCENARIOS[FAILED]
    return Scenario(key=spec.key, label=spec.label, source=SOURCE_TRACE,
                    required=(), forbidden=())


def spec_of(key: str) -> ScenarioSpec:
    return SCENARIOS.get(key, SCENARIOS[OTHER])
