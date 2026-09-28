"""工具调用轨迹。

一轮对话里模型调了哪些工具、传了什么参数、拿到什么结果——这是质检的原料。
采集走的是项目里**已有且已验证**的通道：``monitor_tool`` 中间件写
``request.runtime.context``，下游中间件（``dynamic_prompt``）能读到 ``report``
标记，证明这个 dict 在同一次 ``stream()`` 内是共享的。

本模块刻意只放纯函数：``@wrap_tool_call`` 装饰后得到的是 ``AgentMiddleware``
**实例**而非函数，没法直接调用，所以逻辑必须留在中间件之外才可测。
"""
import itertools

# 模块级计数器：一次模型回复里的多个工具调用是**并发执行**的
# （LangGraph 用 ContextThreadPoolExecutor），list.append 的顺序不等于模型的
# 调用顺序。用单调递增的 seq 标记真实登记次序，读取时据此排序，不依赖 list 顺序。
_seq = itertools.count()


def record_call(trace: list[dict] | None, tool_call: dict) -> dict | None:
    """在工具**执行前**登记一条，返回条目供执行后回填。

    先登记后执行是刻意的：工具抛异常时条目已经在了，回填 ``error`` 即可。
    「模型调了工具但工具炸了」就不会从轨迹里消失。

    :param trace: 调用方持有的列表；为 None 表示本轮不采集（静默返回 None）。
    :return: 待回填的条目；未采集时为 None。
    """
    if trace is None:
        return None

    entry = {
        "seq": next(_seq),
        "id": (tool_call or {}).get("id"),
        "name": (tool_call or {}).get("name"),
        "args": (tool_call or {}).get("args") or {},
        "result": None,
        "error": None,
    }
    trace.append(entry)
    return entry


def finish_call(entry: dict | None, result: str | None = None,
                error: str | None = None) -> None:
    """回填工具的执行结果或错误。entry 为 None（未采集）时静默返回。"""
    if entry is None:
        return
    if error is not None:
        entry["error"] = error
    else:
        entry["result"] = result


def extract_result_text(result) -> str:
    """从工具返回值里取出可读文本。

    ``@wrap_tool_call`` 的 handler 返回 ``ToolMessage``（内容是字符串），
    但也可能是 ``Command`` 或普通值，因此做一层归一。
    """
    if result is None:
        return ""

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多模态分块：只取文本块
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(result, str):
        return result

    text = str(result)
    # Command 之类的对象 str() 出来是一大坨 repr，对质检没有价值
    return "" if text.startswith("Command(") else text


def sorted_calls(trace: list[dict] | None) -> list[dict]:
    """按登记次序返回轨迹条目（不依赖 list 的物理顺序）。"""
    return sorted(trace or [], key=lambda entry: entry.get("seq", 0))


def called_names(trace: list[dict] | None) -> list[str]:
    """本轮实际调用过的工具名，按调用次序。"""
    return [entry["name"] for entry in sorted_calls(trace) if entry.get("name")]


def call_by_name(trace: list[dict] | None, name: str) -> dict | None:
    """取第一个指定名字的调用记录（用于读 ``get_user_id`` 之类的返回值）。"""
    for entry in sorted_calls(trace):
        if entry.get("name") == name:
            return entry
    return None


def calls_by_name(trace: list[dict] | None, name: str) -> list[dict]:
    """取指定名字的全部调用记录（``fetch_external_data`` 可能被调多次）。"""
    return [entry for entry in sorted_calls(trace) if entry.get("name") == name]
