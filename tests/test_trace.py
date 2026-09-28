"""工具调用轨迹的记录与读取。"""
import pytest

from quality.trace import (call_by_name, called_names, calls_by_name,
                           extract_result_text, finish_call, record_call,
                           sorted_calls)


class FakeToolMessage:
    """替身：``@wrap_tool_call`` 的 handler 返回的就是 ToolMessage。"""

    def __init__(self, content):
        self.content = content


# ----------------------------------------------------------------------
# 登记与回填
# ----------------------------------------------------------------------

def test_record_and_finish_a_successful_call():
    trace: list[dict] = []

    entry = record_call(trace, {"id": "c1", "name": "search_book",
                                "args": {"book_name": "三体"}})
    finish_call(entry, result="《三体》可借 3 本")

    assert trace == [{
        "seq": entry["seq"], "id": "c1", "name": "search_book",
        "args": {"book_name": "三体"}, "result": "《三体》可借 3 本", "error": None,
    }]


def test_failed_call_still_leaves_a_record():
    """工具抛异常时条目必须在轨迹里 —— 先登记后执行就是为了这个。

    否则「模型调了工具但工具炸了」会从质检视野里彻底消失，
    等价于把失败算成没发生。
    """
    trace: list[dict] = []

    entry = record_call(trace, {"id": "c1", "name": "fetch_external_data", "args": {}})
    finish_call(entry, error="OperationalError: 连接失败")

    assert len(trace) == 1
    assert trace[0]["error"] == "OperationalError: 连接失败"
    assert trace[0]["result"] is None


def test_record_call_is_silent_when_not_collecting():
    """trace 为 None（调用方不采集）时不得抛异常。"""
    assert record_call(None, {"name": "x", "args": {}}) is None


def test_finish_call_tolerates_none_entry():
    finish_call(None, result="无关紧要")      # 不应抛异常


def test_record_call_tolerates_missing_fields():
    """模型给出的 tool_call 结构不完整时也不该崩。"""
    trace: list[dict] = []

    record_call(trace, {})

    assert trace[0]["name"] is None
    assert trace[0]["args"] == {}


def test_seq_is_monotonic():
    trace: list[dict] = []

    for i in range(3):
        record_call(trace, {"name": f"t{i}", "args": {}})

    seqs = [e["seq"] for e in trace]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3


# ----------------------------------------------------------------------
# 读取次序：不依赖 list 的物理顺序
# ----------------------------------------------------------------------

def test_sorted_calls_orders_by_seq_not_list_position():
    """一次模型回复里的多个工具调用是**并发**执行的，append 顺序可能是乱的
    （LangGraph 用 ContextThreadPoolExecutor）。读取时必须按 seq 还原调用次序。
    """
    trace: list[dict] = []
    a = record_call(trace, {"name": "先调的", "args": {}})
    b = record_call(trace, {"name": "后调的", "args": {}})
    finish_call(a, result="A")
    finish_call(b, result="B")

    # 模拟线程调度导致 list 顺序被打乱
    trace.reverse()

    assert [e["name"] for e in sorted_calls(trace)] == ["先调的", "后调的"]


def test_called_names_in_call_order():
    trace: list[dict] = []
    for name in ("get_user_id", "get_current_month", "fill_context_for_report"):
        record_call(trace, {"name": name, "args": {}})

    assert called_names(trace) == [
        "get_user_id", "get_current_month", "fill_context_for_report"]


def test_helpers_handle_empty_trace():
    assert sorted_calls(None) == []
    assert sorted_calls([]) == []
    assert called_names(None) == []
    assert call_by_name(None, "x") is None
    assert calls_by_name(None, "x") == []


def test_calls_by_name_returns_every_occurrence():
    """fetch_external_data 可能被调用多次，读取时不能只拿第一条。"""
    trace: list[dict] = []
    for uid in ("1005", "1006"):
        record_call(trace, {"name": "fetch_external_data", "args": {"user_id": uid}})

    found = calls_by_name(trace, "fetch_external_data")

    assert [e["args"]["user_id"] for e in found] == ["1005", "1006"]


def test_call_by_name_returns_the_first_match():
    trace: list[dict] = []
    record_call(trace, {"name": "get_user_id", "args": {}, "id": "first"})
    record_call(trace, {"name": "get_user_id", "args": {}, "id": "second"})

    assert call_by_name(trace, "get_user_id")["id"] == "first"


# ----------------------------------------------------------------------
# 结果文本归一
# ----------------------------------------------------------------------

def test_extract_result_text_from_string_content():
    assert extract_result_text(FakeToolMessage("《三体》可借 3 本")) == "《三体》可借 3 本"


def test_extract_result_text_from_multimodal_blocks():
    message = FakeToolMessage([
        {"type": "text", "text": "前半段"},
        {"type": "image_url", "image_url": "…"},     # 非文本块跳过
        {"type": "text", "text": "后半段"},
    ])

    assert extract_result_text(message) == "前半段后半段"


def test_extract_result_text_from_plain_string():
    assert extract_result_text("裸字符串") == "裸字符串"


def test_extract_result_text_from_none():
    assert extract_result_text(None) == ""


def test_extract_result_text_drops_command_repr():
    """Command 之类的对象 str() 出来是一大坨 repr，对质检没有价值，丢弃。"""
    class Command:
        def __repr__(self):
            return "Command(goto='tools', update={...})"

    assert extract_result_text(Command()) == ""


# ----------------------------------------------------------------------
# 与中间件的接线
# ----------------------------------------------------------------------

class _Request:
    """ToolCallRequest 的最小替身。"""

    def __init__(self, context: dict, tool_call: dict):
        self.runtime = type("Runtime", (), {"context": context})()
        self.tool_call = tool_call


def test_monitor_tool_records_into_runtime_context():
    """端到端：中间件从 runtime.context 取 trace 并写入条目。

    ``@wrap_tool_call`` 装饰后得到的是 AgentMiddleware **实例**而非函数，
    不能直接调用，所以这里走它暴露的 ``wrap_tool_call`` 方法（同步方法，
    不返回协程）。
    """
    from agent.tools.middleware import monitor_tool

    trace: list[dict] = []
    request = _Request({"report": False, "tool_calls": trace},
                       {"id": "c1", "name": "get_user_id", "args": {}})

    result = monitor_tool.wrap_tool_call(request, lambda r: FakeToolMessage("1005"))

    assert result.content == "1005"
    assert [(e["name"], e["result"], e["error"]) for e in trace] == [
        ("get_user_id", "1005", None)]


def test_monitor_tool_records_the_failure_then_reraises():
    """工具抛异常：轨迹留痕，异常照常向上抛（不能吞掉）。"""
    from agent.tools.middleware import monitor_tool

    trace: list[dict] = []
    request = _Request({"report": False, "tool_calls": trace},
                       {"id": "c1", "name": "fetch_external_data", "args": {}})

    def _boom(_request):
        raise RuntimeError("连接失败")

    with pytest.raises(RuntimeError, match="连接失败"):
        monitor_tool.wrap_tool_call(request, _boom)

    assert trace[0]["error"] == "RuntimeError: 连接失败"


def test_monitor_tool_still_sets_report_flag_with_trace_absent():
    """不采集轨迹时，原有的 report 标记逻辑必须照常工作。"""
    from agent.tools.middleware import monitor_tool

    context = {"report": False}
    request = _Request(context, {"id": "c1", "name": "fill_context_for_report", "args": {}})

    monitor_tool.wrap_tool_call(request, lambda r: FakeToolMessage("ok"))

    assert context["report"] is True


def test_monitor_tool_tolerates_non_dict_context():
    """context 不是 dict（未来若加了 context_schema）时不得崩在中间件里。"""
    from agent.tools.middleware import monitor_tool

    request = _Request(None, {"id": "c1", "name": "get_user_id", "args": {}})

    result = monitor_tool.wrap_tool_call(request, lambda r: FakeToolMessage("1005"))

    assert result.content == "1005"
