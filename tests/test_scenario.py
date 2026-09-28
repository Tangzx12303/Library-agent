"""场景判定：关键词表必须窄，模糊提问一律落到 other（不适用，不扣分）。"""
import pytest

from quality.scenario import (CATALOG, FAILED, HOURS, KNOWLEDGE, OTHER, REPORT,
                              SEAT, SOURCE_KEYWORD, SOURCE_UNKNOWN,
                              detect_scenario, failed_scenario, spec_of)


@pytest.mark.parametrize("query,expected", [
    # 报告：只认强特征词
    ("根据我的借阅数据进行个性化推荐", REPORT),
    ("帮我生成一份阅读报告", REPORT),
    ("看看我的借阅记录", REPORT),
    # 开放时间
    ("图书馆的开放时间是几点？", HOURS),
    ("周末闭馆吗", HOURS),
    # 座位
    ("自习区还有座位吗", SEAT),
    ("阅览区人多不多", SEAT),
    # 馆藏
    ("帮我查一下《三体》的馆藏情况", CATALOG),
    ("有哪些科幻类的书可以借？", CATALOG),
    ("这本书在馆吗", CATALOG),
    # 知识
    ("图书馆有哪些功能分区？", KNOWLEDGE),
    ("借阅规则是什么", KNOWLEDGE),
    ("办证需要什么材料", KNOWLEDGE),
])
def test_keyword_routing(query, expected):
    assert detect_scenario(query).key == expected


@pytest.mark.parametrize("query", [
    "再推荐几本科幻类的",       # 报告对话的追问，不该重走完整报告流程
    "你好",
    "谢谢",
    "嗯嗯",
    "",
])
def test_ambiguous_queries_fall_through_to_other(query):
    """**模糊提问必须落到 other，且 other 的必需组为空。**

    这是整套指标可信度的地基：只有当误判的代价是「不适用」而不是「0 分」时，
    一张手写的关键词表才敢用来给模型打分。
    """
    scenario = detect_scenario(query)

    assert scenario.key == OTHER
    assert scenario.source == SOURCE_UNKNOWN
    assert scenario.required == ()


def test_open_hours_beats_seat_on_overlap():
    """「自习区的开放时间」两个关键词都命中，开放时间更具体，必须优先。

    若判成 seat，模型正确地调了 get_opening_hours 却没人满足 {check_seat}，
    tool_recall 会变成 0 —— 答对了被判失误。
    """
    assert detect_scenario("自习区的开放时间").key == HOURS


def test_matched_keyword_is_recorded():
    """记下命中的词，规则出问题时能直接看出是哪一条撞的。"""
    assert detect_scenario("图书馆有哪些功能分区？").matched == "功能分区"


# ----------------------------------------------------------------------
# 必需组 / 禁用集
# ----------------------------------------------------------------------

def test_report_requires_the_mandated_chain():
    groups = spec_of(REPORT).required

    assert ("fill_context_for_report",) in groups
    assert ("fetch_external_data",) in groups
    # 最后一组允许用 search_book 核实，属 OR
    assert ("search_books_by_category", "search_book") in groups


@pytest.mark.parametrize("key", [CATALOG, SEAT, HOURS, KNOWLEDGE, OTHER])
def test_non_report_scenarios_forbid_the_report_tool(key):
    """main_prompt.txt 明令：非报告场景绝对不调用 fill_context_for_report。"""
    assert "fill_context_for_report" in spec_of(key).forbidden


def test_report_scenario_forbids_nothing():
    assert spec_of(REPORT).forbidden == ()


def test_other_scenario_has_no_required_groups():
    assert spec_of(OTHER).required == ()


def test_required_flat_expands_or_groups():
    scenario = detect_scenario("有哪些科幻类的书可以借？")

    assert scenario.required_flat == {"search_book", "search_books_by_category"}


# ----------------------------------------------------------------------
# 轨迹佐证
# ----------------------------------------------------------------------

def test_trace_does_not_override_keyword_verdict():
    """模型误调 fill_context_for_report 时，**不能**把场景改判成报告来掩盖问题。

    轨迹里出现这个工具是「模型自己声明要报告」的事实，但若关键词判定是别的场景，
    就该让「未误触发报告流程」那条检查报警 —— 那才是我们要看见的缺陷。
    """
    trace = [{"name": "fill_context_for_report", "args": {}, "seq": 0}]

    scenario = detect_scenario("图书馆有哪些功能分区？", trace)

    assert scenario.key == KNOWLEDGE
    assert "fill_context_for_report" in scenario.forbidden


def test_failed_scenario_is_metric_free():
    """异常中断轮不参与任何比率统计。"""
    scenario = failed_scenario()

    assert scenario.key == FAILED
    assert scenario.required == ()
    assert scenario.forbidden == ()


def test_unknown_spec_falls_back_to_other():
    assert spec_of("不存在的场景").key == OTHER
