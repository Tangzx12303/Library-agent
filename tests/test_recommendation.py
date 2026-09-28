"""推荐书目识别 —— 本项目质检里最容易写错的一块。"""
import pytest

from quality.recommendation import (CLOSING_FOOTER, analyze_recommendations,
                                    extract_candidates, looks_like_raw_tool_dump)


def _call(name, result):
    return {"seq": 0, "name": name, "args": {}, "result": result, "error": None}


# 一份真实的工具返回（search_books_by_category）
CATEGORY_RESULT = """类别「绘画/音乐」下当前在架可借图书共 3 种：
1. 《西方美术史》作者：张敢，索书号：J110.9/0043，馆藏位置：3楼社会科学阅览区，可借 3 本
2. 《聆听音乐》作者：克雷格·莱特，索书号：J605/0021，馆藏位置：3楼社会科学阅览区，可借 3 本
3. 《音乐是怎样算成的》作者：伊莱·马奥尔，索书号：J60-05/0004，馆藏位置：3楼社会科学阅览区，可借 2 本
"""

# 一份真实的工具返回（search_book）—— 含一本已借空的
SEARCH_RESULT = """《艺术的故事》作者：贡布里希，类别：绘画，索书号：J110.9/0042，馆藏位置：3楼社会科学阅览区，可借 0 本 / 共 4 本，状态：已借空
《梵高手稿》作者：文森特·梵高，类别：绘画，索书号：J231/0013，馆藏位置：3楼社会科学阅览区，可借 0 本 / 共 3 本，状态：已借空"""


# ----------------------------------------------------------------------
# 候选集：从工具轨迹提取「在架可借」
# ----------------------------------------------------------------------

def test_candidates_only_include_available_books():
    """可借数为 0 的不进候选集 —— 与 SQL 层的 available_stock > 0 同源。"""
    trace = [_call("search_block".replace("block", "book"), SEARCH_RESULT)]

    assert extract_candidates(trace) == []


def test_candidates_from_category_result():
    trace = [_call("search_books_by_category", CATEGORY_RESULT)]

    assert extract_candidates(trace) == ["西方美术史", "聆听音乐", "音乐是怎样算成的"]


def test_candidates_ignore_unrelated_tools():
    """RAG / 借阅记录都不该混进候选集。"""
    trace = [
        _call("rag_summarize", "《图书馆学概论》是一本专业书"),
        _call("fetch_external_data", '{"在借图书": "《艺术的故事》《梵高手稿》"}'),
    ]

    assert extract_candidates(trace) == []


def test_candidates_skip_entries_without_result():
    """工具失败（只有 error 没有 result）时不该崩。"""
    trace = [{"seq": 0, "name": "search_books_by_category", "args": {},
              "result": None, "error": "OperationalError"}]

    assert extract_candidates(trace) == []


# ----------------------------------------------------------------------
# 核心回归：否定句不得污染推荐
# ----------------------------------------------------------------------

def test_borrowed_out_book_in_a_negation_is_not_a_recommendation():
    """**这是本模块存在的理由。**

    report_prompt.txt 明确要求模型写「该书当前已借空，建议改借同类别可借的书」。
    若直接正则抓《》，已借空的《艺术的故事》会被当成一次推荐，再对着正确排除了
    它的候选集算成假阳性 —— 模型严格遵守提示词却被判失误。
    """
    trace = [
        _call("search_books_by_category", CATEGORY_RESULT),
        _call("search_book", SEARCH_RESULT),
    ]
    response = (
        "## 图书借阅个性化推荐报告\n\n"
        "您当前在借的《艺术的故事》已借空，建议改借同类别下可借的图书。\n\n"
        "## 推荐书目\n"
        "1. 《西方美术史》作者：张敢，索书号：J110.9/0043，可借 3 本\n"
        "2. 《聆听音乐》作者：克雷格·莱特，索书号：J605/0021，可借 3 本\n\n"
        + CLOSING_FOOTER + "，实际库存以到馆时为准。"
    )

    analysis = analyze_recommendations(response, trace)

    assert "艺术的故事" in analysis.negated
    assert "艺术的故事" not in analysis.recommended, "否定句里的书名被误判成推荐"
    assert set(analysis.recommended) == {"西方美术史", "聆听音乐"}
    assert analysis.hallucinated == []


def test_negation_and_recommendation_in_the_same_line():
    """同一行既否定又推荐 —— 所以必须按**子句**切分，不能按行。"""
    trace = [_call("search_books_by_category", CATEGORY_RESULT)]
    response = "《艺术的故事》已借空，建议改借《西方美术史》，该书可借 3 本。"

    analysis = analyze_recommendations(response, trace)

    assert analysis.negated == ["艺术的故事"]
    assert "西方美术史" in analysis.recommended


@pytest.mark.parametrize("phrase", [
    "已借空", "已借出", "暂不可借", "暂无在架", "无法借阅", "已经借完",
    "可借 0 本", "可借数量为 0", "没有可借复本",
])
def test_all_negation_phrasings(phrase):
    response = f"《沙丘》{phrase}，请改借其它图书。"

    analysis = analyze_recommendations(response, [])

    assert "沙丘" in analysis.negated
    assert analysis.recommended == []


# ----------------------------------------------------------------------
# 编造书目
# ----------------------------------------------------------------------

def test_hallucinated_book_is_detected():
    trace = [_call("search_books_by_category", CATEGORY_RESULT)]
    response = "## 推荐书目\n- 《不存在的书》作者：某人，可借 5 本，索书号：X1/0001"

    analysis = analyze_recommendations(response, trace)

    assert analysis.hallucinated == ["不存在的书"]
    assert analysis.in_candidates == []


def test_hallucination_is_detected_even_when_diluted_by_good_books():
    """这正是要把「无编造书目」做成二元检查而不是比率的原因 ——

    一本编造的书混在若干本好书里，准确率只掉几个点，信号会被稀释掉。
    """
    trace = [_call("search_books_by_category", CATEGORY_RESULT)]
    response = (
        "## 推荐书目\n"
        "1. 《西方美术史》索书号：J110.9/0043，可借 3 本\n"
        "2. 《聆听音乐》索书号：J605/0021，可借 3 本\n"
        "3. 《音乐是怎样算成的》索书号：J60-05/0004，可借 2 本\n"
        "4. 《西方音乐通史》索书号：J609/9999，可借 4 本\n"
    )

    analysis = analyze_recommendations(response, trace)

    assert len(analysis.in_candidates) == 3
    assert analysis.hallucinated == ["西方音乐通史"]


# ----------------------------------------------------------------------
# 分类的兜底规则
# ----------------------------------------------------------------------

def test_bullet_list_under_recommend_heading_counts_as_recommended():
    """书名和可借数被逗号分开时的兜底：靠标题判定。"""
    response = "### 推荐书目\n- 《西方美术史》\n- 《聆听音乐》\n"

    analysis = analyze_recommendations(response, [])

    assert set(analysis.recommended) == {"西方美术史", "聆听音乐"}


def test_candidate_under_closing_note_counts_as_recommended():
    """落在候选集内 + 回答含提示词要求的收尾声明 → 算推荐。"""
    trace = [_call("search_books_by_category", CATEGORY_RESULT)]
    response = "为您挑选：\n《西方美术史》\n\n" + CLOSING_FOOTER

    analysis = analyze_recommendations(response, trace)

    assert "西方美术史" in analysis.recommended


# 一次真实的模型输出：它把可借书目排成了 markdown 表格。
# 单元格里写的是「4 本」而不是「可借 4 本」，所以只靠「可借 N」的规则会整表漏判，
# 把一次完全正确的推荐判成召回率 0。
REAL_TABLE_RESPONSE = """已为您查到科幻类当前在架可借的书目，全部位于**2楼文学借阅区**，可直接前往取阅：

| 序号 | 书名 | 作者 | 索书号 | 可借数量 |
|---|---|---|---|---|
| 1 | 《流浪地球》 | 刘慈欣 | I247.5/1201 | 4 本 |
| 2 | 《三体》 | 刘慈欣 | I247.5/1200 | 3 本 |
| 3 | 《基地》 | 艾萨克·阿西莫夫 | I712.45/0331 | 2 本 |
| 4 | 《银河系漫游指南》 | 道格拉斯·亚当斯 | I561.45/0218 | 1 本 |

**借阅小建议：**
- 📖 **《三体》《流浪地球》**：刘慈欣代表作，中文硬科幻入门首选；
- 🌌 **《基地》**：阿西莫夫经典系列开篇，偏重"心理史学"与文明演化的宏大构想；
"""

SCIENCE_FICTION_RESULT = """类别「科幻」下当前在架可借图书共 4 种：
1. 《流浪地球》作者：刘慈欣，索书号：I247.5/1201，馆藏位置：2楼文学借阅区，可借 4 本
2. 《三体》作者：刘慈欣，索书号：I247.5/1200，馆藏位置：2楼文学借阅区，可借 3 本
3. 《基地》作者：艾萨克·阿西莫夫，索书号：I712.45/0331，馆藏位置：2楼文学借阅区，可借 2 本
4. 《银河系漫游指南》作者：道格拉斯·亚当斯，索书号：I561.45/0218，馆藏位置：2楼文学借阅区，可借 1 本
"""


def test_markdown_table_rows_count_as_recommendations():
    """回归：模型用表格排版荐书时必须识别出来。

    这条来自一次真实的对话 —— 模型答得完全正确（列出了全部 4 本在架可借的科幻
    书），却因为表格单元格里写的是「4 本」而非「可借 4 本」被判成「什么都没推荐」，
    召回率记成 0。
    """
    trace = [_call("search_books_by_category", SCIENCE_FICTION_RESULT)]

    analysis = analyze_recommendations(REAL_TABLE_RESPONSE, trace)

    assert set(analysis.recommended) == {"流浪地球", "三体", "基地", "银河系漫游指南"}
    assert analysis.hallucinated == []
    assert analysis.in_candidates == analysis.recommended


def test_bullet_list_with_candidates_counts_as_recommendations():
    """加粗的列表项同样是推荐形态（上例的「借阅小建议」段落）。"""
    trace = [_call("search_books_by_category", SCIENCE_FICTION_RESULT)]

    analysis = analyze_recommendations(
        "- **《三体》《流浪地球》**：刘慈欣代表作，中文硬科幻入门首选；", trace)

    assert set(analysis.recommended) == {"三体", "流浪地球"}


def test_table_row_with_hallucinated_book_is_still_flagged():
    """表格形态不能成为编造书目的挡箭牌。"""
    trace = [_call("search_books_by_category", SCIENCE_FICTION_RESULT)]
    response = "| 序号 | 书名 | 索书号 |\n|---|---|---|\n| 1 | 《海伯利安》 | I712.45/9999 |\n"

    analysis = analyze_recommendations(response, trace)

    assert analysis.hallucinated == ["海伯利安"]


def test_plain_lookup_sentence_is_not_a_recommendation():
    """单本书的查证式回答不算推荐，指标应为不适用而不是低召回。

    判据来自候选集的自然范围：查单本书时模型会调 search_book，它只返回那一本，
    候选集因此只有一本 —— 即使这里算成推荐，召回率也是 1/1。
    """
    trace = [_call("search_book", "《三体》作者：刘慈欣，索书号：I247.5/1200，可借 3 本 / 共 5 本")]

    analysis = analyze_recommendations("《三体》当前在馆，可借 3 本，位于2楼文学借阅区。", trace)

    assert analysis.recommended == ["三体"]
    assert analysis.candidates == ["三体"]


def test_incidental_mention_is_unclassified():
    """与推荐无关的提及不该计入指标 —— 宁可少算，不可错算。"""
    response = "图书馆收藏有《四库全书》的影印本，可供馆内阅览。"

    analysis = analyze_recommendations(response, [])

    assert analysis.recommended == []
    assert analysis.negated == []
    assert "四库全书" in analysis.unclassified


def test_same_book_both_recommended_and_negated_is_flagged():
    """自相矛盾要被记下来，由检查项去报，而不是在这里悄悄合并掉。"""
    trace = [_call("search_books_by_category", CATEGORY_RESULT)]
    response = (
        "## 推荐书目\n"
        "《西方美术史》可借 3 本，索书号：J110.9/0043\n"
        "注意：《西方美术史》已借空，可能无法借到。\n"
    )

    analysis = analyze_recommendations(response, trace)

    assert "西方美术史" in analysis.recommended
    assert "西方美术史" in analysis.negated
    assert analysis.both_recommended_and_negated == ["西方美术史"]


def test_analysis_is_empty_for_blank_response():
    analysis = analyze_recommendations("", None)

    assert analysis.recommended == []
    assert analysis.candidates == []


def test_titles_are_normalized():
    """书名里的空格/换行不该造成重复项。"""
    response = "## 推荐书目\n《西方 美术史》可借 3 本"

    analysis = analyze_recommendations(response, [])

    assert analysis.recommended == ["西方美术史"]


# ----------------------------------------------------------------------
# 工具原始返回的识别
# ----------------------------------------------------------------------

@pytest.mark.parametrize("marker", ["当前在架可借图书共", '"偏好类别"', "参考元数据"])
def test_raw_tool_dump_is_detected(marker):
    assert looks_like_raw_tool_dump(f"好的，结果如下：{marker} ……") == marker


def test_clean_response_is_not_flagged():
    assert looks_like_raw_tool_dump("根据您的借阅数据，推荐《西方美术史》。") is None
