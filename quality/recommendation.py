"""从回答里识别「推荐了哪些书」，并把它和「只是提到」区分开。

## 为什么不能直接正则抓《》里的书名

``prompts/report_prompt.txt`` **明确指示**模型写这样一句话：

> 读者历史借阅过的书若当前已借空（可借数量为 0），只能说明「该书当前已借空」
> 并推荐同类别下可借的替代书目

如果直接抓书名，《艺术的故事》（种子数据里 ``available_stock = 0``）会被从这句
**否定句**里抓出来当成一次推荐，再对着正确排除了它的候选集算成**假阳性**——
**模型严格遵守了提示词，却被判失误**。

所以必须做两件事：**按子句切分**（一行里完全可能既否定又推荐），以及
**识别否定语气**。

## 候选集为什么从轨迹里取，而不是重新查库

候选集是「模型实际拿到的在架可借书目」，它由 ``search_books_by_category`` 的
SQL（带 ``available_stock > 0``）过滤过。质检要判断的是「模型有没有用好喂给它的
数据」，所以必须用它**实际收到**的那份，而不是评测时另查一次数据库——否则一旦
库存在此期间变动，就会拿一把变化的尺子去量一个固定的回答。
"""
import re
from dataclasses import dataclass, field

from quality.trace import sorted_calls

# 书名：《…》。长度上限 60 是防御性的，避免正文里的书名号配对错乱时吞掉一大段
TITLE_RE = re.compile(r"《([^》\n]{1,60})》")

# 工具返回里「可借 N 本」——N > 0 才算在架可借
AVAILABILITY_RE = re.compile(r"可借\s*([0-9]+)\s*本")

# 否定语气：说明「这本书现在借不到」
NEGATION_RE = re.compile(
    r"已借空|已借出|均已借出|不可借|暂不可借|暂无在架|无法借阅|借完|借罄"
    r"|可借\s*0\s*本|可借数量为\s*0|没有可借|无可借复本"
)

# 明确的可借宣告
AVAILABLE_RE = re.compile(r"可借\s*[1-9]")

# 列表 / 表格形态：模型常把可借书目排版成 markdown 表格，单元格里写「4 本」
# 而不是「可借 4 本」，只靠 AVAILABLE_RE 会整表漏判。
_TABLE_ROW_RE = re.compile(r"\|\s*[^|\n]*\|")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.、)]\s)")

# 索书号，形如 J110.9/0043、TP311.52/0140、I247.5/1200。
# report_prompt.txt 要求推荐的每一本都标注索书号，因此它同时是一个可靠的
# 「在向读者交付可取书目」信号 —— 用来把结构化列表里的**编造书目**也识别出来。
CALL_NUMBER_RE = re.compile(r"[A-Z]{1,3}\d+(?:\.\d+)?/\d+")

# 章节标题里出现这些词，其下的书名视为推荐
HEADING_RE = re.compile(r"推荐|建议阅读|延伸阅读|为你推荐|推荐书目|馆藏推荐")

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")
_BOLD_ONLY_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*$")

# 子句切分：句末标点 + 逗号 + markdown 表格分隔。
# **逗号必须切**，否则「《艺术的故事》已借空，建议改借《西方美术史》」会整体
# 落进否定分支，把本该推荐的《西方美术史》一起否掉。
_CLAUSE_SPLIT_RE = re.compile(r"[。！？；！？，,\n\r|]+")

# report_prompt.txt 要求报告末尾必须附的那句话
CLOSING_FOOTER = "以上推荐书目均已确认在馆且可借"

# 用于判定「模型把工具的原始返回直接倒出来了」
RAW_TOOL_MARKERS = ("当前在架可借图书共", '"偏好类别"', '"月度借阅量"', "参考元数据")


@dataclass
class RecommendationAnalysis:
    """一轮对话里的书目识别结果。"""

    candidates: list[str] = field(default_factory=list)   # 工具给过的在架可借书目
    recommended: list[str] = field(default_factory=list)  # 判定为「推荐」的书名
    negated: list[str] = field(default_factory=list)      # 被说明为借不到的书名
    unclassified: list[str] = field(default_factory=list)  # 其余提及，不计入指标
    heading_hits: list[str] = field(default_factory=list)  # 命中的推荐章节标题

    @property
    def in_candidates(self) -> list[str]:
        """推荐且确实在候选集里的（= 好推荐）。"""
        pool = set(self.candidates)
        return [t for t in self.recommended if t in pool]

    @property
    def hallucinated(self) -> list[str]:
        """推荐了但候选集里没有的（= 编造，或推荐了已借空的）。"""
        pool = set(self.candidates)
        return [t for t in self.recommended if t not in pool]

    @property
    def both_recommended_and_negated(self) -> list[str]:
        """同一本书既被推荐又被说成借不到 —— 自相矛盾。"""
        return [t for t in self.recommended if t in set(self.negated)]


def _normalize(title: str) -> str:
    return re.sub(r"\s+", "", title or "")


def extract_candidates(trace: list[dict] | None) -> list[str]:
    """从工具轨迹里取出「在架可借」的书名。

    ``search_book`` 与 ``search_books_by_category`` 的返回都是逐行的
    ``《书名》…可借 N 本…``，所以用同一条规则提取：**该行可借数大于 0**。
    已借空的条目（``可借 0 本``）自然被排除——这与 SQL 层的约束同源。
    """
    titles: list[str] = []
    for entry in sorted_calls(trace):
        if entry.get("name") not in ("search_book", "search_books_by_category"):
            continue
        for line in (entry.get("result") or "").splitlines():
            match = TITLE_RE.search(line)
            if not match:
                continue
            availability = AVAILABILITY_RE.search(line)
            if availability and int(availability.group(1)) > 0:
                titles.append(_normalize(match.group(1)))
    return _dedupe(titles)


def _dedupe(items) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _heading_of(line: str) -> str | None:
    """把 markdown 标题行或整行加粗的「伪标题」提取出来。"""
    for pattern in (_MD_HEADING_RE, _BOLD_ONLY_RE):
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def analyze_recommendations(response: str,
                            trace: list[dict] | None = None) -> RecommendationAnalysis:
    """分析回答里的书目提及，区分「推荐」「否定」「未分类」。

    分类规则（按优先级，先命中者胜）：

    1. 子句含**否定语气** → negated
    2. 子句含 ``可借 N``（N>0） → recommended
    3. **所在行**含 ``可借 N`` → recommended（书名和可借数常被逗号分开）
    4. 最近的上级标题含「推荐」 → recommended
    5. 书名在候选集内，且落在**列表项或表格行**里 → recommended
    6. 落在列表项或表格行里，且该行含**索书号** → recommended
    7. 书名在候选集内，且回答含提示词要求的收尾声明 → recommended
    8. 其余 → unclassified（不计入任何指标）

    规则 5–6 都是补出来的：

    - 模型很喜欢把可借书目排成 markdown 表格，单元格里写「4 本」而不是「可借 4 本」，
      只靠前三条会把整张表漏判成「什么都没推荐」—— 一条真实的科幻类推荐因此被判成
      召回率 0，而模型答得完全正确。
    - 规则 6 **不要求书名在候选集内**，这一点是必须的：编造的书恰恰不在候选集里，
      若把结构化列表的判据绑在候选集上，「表格里的假书」这个最该抓的情形反而会漏掉。
      用索书号而不是「只要是列表就算」做判据，则是因为它会误伤另一类真实回答 ——
      「您当前在借：- 《艺术的故事》」这种列表不该被算成推荐。

    同一书名在不同子句里可能既被推荐又被否定（规则 1 与 2 分别命中），两个集合
    都会记下来 —— 由「同书未同时推荐与否定」这条检查项去报缺陷，而不是在这里
    悄悄合并掉。
    """
    analysis = RecommendationAnalysis(candidates=extract_candidates(trace))

    text = response or ""
    has_closing = CLOSING_FOOTER in text
    candidate_pool = set(analysis.candidates)

    recommended: list[str] = []
    negated: list[str] = []
    unclassified: list[str] = []
    headings: list[str] = []

    current_heading = ""
    for line in text.splitlines():
        heading = _heading_of(line)
        if heading:
            current_heading = heading
            if HEADING_RE.search(heading):
                headings.append(heading)
            # 标题行本身通常不含书名，继续处理该行剩余部分也无妨
        line_has_availability = bool(AVAILABLE_RE.search(line))
        line_is_list = bool(_TABLE_ROW_RE.search(line) or _LIST_ITEM_RE.match(line))
        line_has_call_number = bool(CALL_NUMBER_RE.search(line))

        for clause in _CLAUSE_SPLIT_RE.split(line):
            if not clause.strip():
                continue
            titles = _dedupe(_normalize(m.group(1)) for m in TITLE_RE.finditer(clause))
            if not titles:
                continue

            is_negated = bool(NEGATION_RE.search(clause))
            is_available = bool(AVAILABLE_RE.search(clause))
            under_recommend_heading = bool(HEADING_RE.search(current_heading))

            for title in titles:
                if is_negated:
                    negated.append(title)
                elif is_available or line_has_availability:
                    recommended.append(title)
                elif under_recommend_heading:
                    recommended.append(title)
                elif title in candidate_pool and line_is_list:
                    recommended.append(title)
                elif line_is_list and line_has_call_number:
                    recommended.append(title)
                elif title in candidate_pool and has_closing:
                    recommended.append(title)
                else:
                    unclassified.append(title)

    analysis.recommended = _dedupe(recommended)
    analysis.negated = _dedupe(negated)
    analysis.unclassified = _dedupe(
        t for t in unclassified
        if t not in set(analysis.recommended) and t not in set(analysis.negated)
    )
    analysis.heading_hits = _dedupe(headings)
    return analysis


def looks_like_raw_tool_dump(response: str) -> str | None:
    """回答里是否直出了工具原始返回。返回命中的特征串，未命中返回 None。"""
    text = response or ""
    for marker in RAW_TOOL_MARKERS:
        if marker in text:
            return marker
    return None
