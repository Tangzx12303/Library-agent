"""端到端：一轮对话的追踪数据 → 质检报告。"""
import json

import pytest

from quality.evaluator import evaluate_turn
from quality.scenario import CATALOG, KNOWLEDGE, OTHER, REPORT

CATEGORY_RESULT = """类别「绘画/音乐」下当前在架可借图书共 4 种：
1. 《西方美术史》作者：张敢，索书号：J110.9/0043，可借 3 本
2. 《聆听音乐》作者：克雷格·莱特，索书号：J605/0021，可借 3 本
3. 《音乐是怎样算成的》作者：伊莱·马奥尔，索书号：J60-05/0004，可借 2 本
4. 《西方音乐史》作者：保罗·亨利·朗，索书号：J609.5/0008，可借 1 本
"""
FOOTER = "以上推荐书目均已确认在馆且可借，实际库存以到馆时为准。"


def _trace(*entries):
    return [{"seq": i, "name": name, "args": args or {}, "result": result, "error": None}
            for i, (name, args, result) in enumerate(entries)]


REPORT_TRACE = _trace(
    ("get_user_id", {}, "1005"),
    ("get_current_month", {}, "2025-04"),
    ("fill_context_for_report", {}, "fill_context_for_report已调用"),
    ("fetch_external_data", {"user_id": "1005", "month": "2025-04"},
     json.dumps({"偏好类别": "艺术读者 | 绘画/音乐", "月度借阅量": "月借4册"},
                ensure_ascii=False)),
    ("search_books_by_category", {"category": "绘画/音乐"}, CATEGORY_RESULT),
)

GOOD_REPORT = (
    "## 图书借阅个性化推荐报告\n\n"
    "您当前在借的《艺术的故事》已借空，建议改借同类别下可借的图书。\n\n"
    "## 推荐书目\n"
    "1. 《西方美术史》索书号：J110.9/0043，可借 3 本\n"
    "2. 《聆听音乐》索书号：J605/0021，可借 3 本\n\n" + FOOTER
)


# ----------------------------------------------------------------------
# 正常路径
# ----------------------------------------------------------------------

def test_good_report_scores_well():
    report = evaluate_turn(
        query="根据我的借阅数据进行个性化推荐",
        response=GOOD_REPORT, trace=REPORT_TRACE,
        reader_id="1005", model_id="qwen3.8-max", latency_ms=4200,
    )

    assert report.scenario.key == REPORT
    assert report.rec_precision == 1.0          # 推荐的两本都在候选集里
    assert report.rec_recall == pytest.approx(0.5)   # 候选 4 本，推了 2 本
    assert report.tool_recall == 1.0            # 固定工具链走完
    assert report.tool_precision == 1.0
    assert report.hallucinated == []
    assert report.pass_rate == 1.0
    assert report.model_id == "qwen3.8-max"
    assert report.latency_ms == 4200


def test_borrowed_out_book_does_not_hurt_precision():
    """回归：提示词要求模型说明「该书已借空」，这不该被算成推荐失败。

    这是本项目质检里最容易被写错的一处 —— 直接抓《》会把否定句里的书名
    当成推荐，再对着正确排除了它的候选集算成假阳性。
    """
    report = evaluate_turn(
        query="根据我的借阅数据进行个性化推荐",
        response=GOOD_REPORT, trace=REPORT_TRACE, reader_id="1005",
    )

    assert "艺术的故事" in report.negated
    assert "艺术的故事" not in report.recommended
    assert report.rec_precision == 1.0


# ----------------------------------------------------------------------
# 问题路径
# ----------------------------------------------------------------------

def test_hallucinated_books_are_caught():
    bad = ("## 推荐书目\n"
           "1. 《西方美术史》索书号：J110.9/0043，可借 3 本\n"
           "2. 《绘画心理学》索书号：B84/9999，可借 5 本\n" + FOOTER)

    report = evaluate_turn("根据我的借阅数据进行个性化推荐", bad,
                           trace=REPORT_TRACE, reader_id="1005")

    assert report.hallucinated == ["绘画心理学"]
    assert report.rec_precision == pytest.approx(0.5)
    assert any(c.name == "无编造书目" and c.passed is False for c in report.checks)


def test_incomplete_report_chain_lowers_tool_recall():
    partial = [e for e in REPORT_TRACE if e["name"] != "search_books_by_category"]

    report = evaluate_turn("根据我的借阅数据推荐", GOOD_REPORT,
                           trace=partial, reader_id="1005")

    assert report.tool_recall == pytest.approx(2 / 3)
    assert ["search_books_by_category", "search_book"] in report.unsatisfied_groups


def test_unauthorized_read_is_flagged():
    trace = _trace(
        ("get_user_id", {}, "1005"),
        ("fill_context_for_report", {}, "ok"),
        ("fetch_external_data", {"user_id": "1006", "month": "2025-04"}, "{}"),
        ("search_books_by_category", {}, CATEGORY_RESULT),
    )

    report = evaluate_turn("根据我的借阅数据推荐", GOOD_REPORT,
                           trace=trace, reader_id="1005")

    assert any(c.name == "无越权查询" and c.passed is False for c in report.checks)


# ----------------------------------------------------------------------
# 不适用：纯知识问答
# ----------------------------------------------------------------------

def test_knowledge_turn_leaves_recommendation_metrics_null():
    """纯知识问答没有推荐 —— 指标必须是 None（落库为 NULL），不是 0。

    这是管理员图表可信度的前提：MySQL 的 AVG() 跳过 NULL，这些轮次不会
    把推荐 F1 的均值往下拽。
    """
    trace = _trace(("rag_summarize", {"query": "功能分区"},
                    "【参考资料1】图书馆分为…"))

    report = evaluate_turn("图书馆有哪些功能分区？", "图书馆分为借阅区、阅览区…",
                           trace=trace, reader_id="1005")

    assert report.scenario.key == KNOWLEDGE
    assert report.rec_precision is None
    assert report.rec_recall is None
    assert report.rec_f1 is None
    assert report.tool_recall == 1.0


def test_other_scenario_has_no_tool_metric():
    """场景判不出来时，工具指标不适用 —— 误判不该产生扣分。"""
    report = evaluate_turn("你好", "您好，请问需要什么帮助？", trace=[], reader_id="1005")

    assert report.scenario.key == OTHER
    assert report.tool_recall is None
    assert report.tool_f1 is None


def test_pass_rate_denominator_counts_only_determined_checks():
    """通过率分母是「判定过的项数」，不是检查项总数。

    一轮普通问答只判定 5–8 项，拿固定总数当分母会让通过率随场景漂移。
    """
    report = evaluate_turn("你好", "您好", trace=[], reader_id="1005")

    assert report.determined_count < len(report.checks)
    assert report.not_applicable_count > 0
    assert report.pass_rate == report.passed_count / report.determined_count


# ----------------------------------------------------------------------
# 落库形态
# ----------------------------------------------------------------------

def test_to_row_is_serializable_and_truncates_query():
    report = evaluate_turn("问" * 900, "答", trace=[], reader_id="1005",
                           model_id="m", latency_ms=10)

    row = report.to_row()

    assert len(row["query"]) == 512
    assert isinstance(row["detail_json"], str)
    assert json.loads(row["detail_json"])["tools_called"] == []


def test_to_row_keeps_nulls_as_none():
    """NULL 必须原样传下去，不能变成 0。"""
    report = evaluate_turn("你好", "您好", trace=[], reader_id="1005")

    row = report.to_row()

    assert row["rec_precision"] is None
    assert row["rec_recall"] is None
    assert row["tool_recall"] is None


def test_to_detail_does_not_carry_raw_tool_output():
    """只存派生事实，不存工具返回的原始文本 —— 否则表会涨得很快。"""
    report = evaluate_turn("根据我的借阅数据进行个性化推荐", GOOD_REPORT,
                           trace=REPORT_TRACE, reader_id="1005")

    detail = json.dumps(report.to_detail(), ensure_ascii=False)

    assert "可借 3 本" not in detail
    assert "偏好类别" not in detail
    assert "西方美术史" in detail       # 书名这类派生事实要保留


def test_to_compact_excludes_long_lists():
    """session_state 里逐条重放用的精简版不该塞进候选书目。"""
    report = evaluate_turn("根据我的借阅数据进行个性化推荐", GOOD_REPORT,
                           trace=REPORT_TRACE, reader_id="1005")

    compact = report.to_compact()

    assert "candidates" not in compact
    assert compact["rec_precision"] == 1.0
    assert compact["scenario_label"] == "推荐报告"


# ----------------------------------------------------------------------
# 健壮性
# ----------------------------------------------------------------------

def test_evaluate_turn_survives_missing_trace():
    report = evaluate_turn("你好", "您好", trace=None, reader_id="1005")

    assert report.tools_called == []
    assert report.pass_rate is not None


def test_evaluate_turn_survives_empty_response():
    report = evaluate_turn("你好", "", trace=[], reader_id="1005")

    assert any(c.name == "回答非空" and c.passed is False for c in report.checks)


def test_duplicate_tool_calls_are_preserved():
    trace = _trace(
        ("search_book", {"book_name": "三体"}, "《三体》可借 3 本 / 共 5 本"),
        ("search_book", {"book_name": "沙丘"}, "《沙丘》可借 0 本 / 共 3 本"),
    )

    report = evaluate_turn("帮我查一下《三体》的馆藏情况", "《三体》可借 3 本",
                           trace=trace, reader_id="1005")

    assert report.scenario.key == CATALOG
    assert report.tools_called == ["search_book", "search_book"]
    assert report.candidates == ["三体"]      # 已借空的《沙丘》不进候选集
