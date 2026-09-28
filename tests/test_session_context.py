"""会话上下文：整个对话内读者身份与月份固定，且并发会话之间互相隔离。"""
import threading

from agent.tools import agent_tools
from agent.tools.agent_tools import (apply_session_context, get_current_month,
                                     get_user_id, new_session_context)


# ----------------------------------------------------------------------
# 对话内固定
# ----------------------------------------------------------------------

def test_user_id_is_stable_within_a_conversation(bound_reader):
    """同一个对话里反复取，必须永远是同一个值。

    这是原来「每次调用都重新随机」那个 bug 的回归测试 —— 它曾导致同一次提问中
    两次 get_user_id 返回不同读者，生成的报告前后数据对不上。
    """
    bound_reader("1005")

    assert len({get_user_id.invoke({}) for _ in range(5)}) == 1
    assert get_user_id.invoke({}) == "1005"


def test_month_is_stable_within_a_conversation(bound_reader):
    bound_reader("1005", month="2025-04")

    assert len({get_current_month.invoke({}) for _ in range(5)}) == 1
    assert get_current_month.invoke({}) == "2025-04"


def test_new_session_context_honours_explicit_reader_id():
    """登录读者的真实身份优先，只有未指定时才随机兜底。"""
    assert new_session_context("1005")["user_id"] == "1005"


def test_new_session_context_falls_back_to_a_real_reader():
    """脱离登录页单独调用时的兜底：随机取一位种子读者。"""
    user_id = new_session_context()["user_id"]

    assert user_id in {f"{1000 + i}" for i in range(1, 11)}


def test_rebinding_switches_identity(bound_reader):
    """重新注入（清空对话 / 换账号）必须真的换掉身份。"""
    bound_reader("1001")
    assert get_user_id.invoke({}) == "1001"

    bound_reader("1002")
    assert get_user_id.invoke({}) == "1002"


# ----------------------------------------------------------------------
# 并发隔离
# ----------------------------------------------------------------------

def test_context_is_isolated_between_threads():
    """并发会话不得串味。

    项目的 Agent 被 ``@st.cache_resource`` 全局共享，所以绝不能用模块级全局变量
    存身份 —— A 用户设成 1003、B 用户同时提问就会读到 A 的值，是数据越权级别的
    Bug。contextvars 的作用域绑定执行上下文，新线程拿到的是独立的上下文。
    """
    main_id = "1001"
    other_thread_id = "2002"
    seen: dict[str, str] = {}
    ready = threading.Barrier(2)

    apply_session_context({"user_id": main_id, "month": "2025-04"})

    def worker():
        agent_tools.apply_session_context({"user_id": other_thread_id, "month": "2025-07"})
        ready.wait()
        seen["thread"] = get_user_id.invoke({})
        seen["thread_month"] = get_current_month.invoke({})

    t = threading.Thread(target=worker)
    t.start()
    ready.wait()
    seen["main"] = get_user_id.invoke({})
    t.join()

    assert seen["thread"] == other_thread_id
    assert seen["thread_month"] == "2025-07"
    assert seen["main"] == main_id, "子线程的注入污染了主线程的会话身份"


# ----------------------------------------------------------------------
# contextvars 在工具调用里的传播语义（踩过一次，钉死）
# ----------------------------------------------------------------------

def test_context_is_readable_inside_tool_invocation(bound_reader):
    """**读**是传播的：绑定后经工具接口调用也能读到。

    这是生产路径依赖的语义。LangGraph 用 ContextThreadPoolExecutor 执行工具，
    它在线程提交时**复制**父上下文，所以子线程看得到父线程注入的值。
    """
    bound_reader("1007")

    assert get_user_id.invoke({}) == "1007"


def test_context_write_does_not_propagate_out_of_tool_invocation(bound_reader):
    """**写**不传播：``StructuredTool.invoke()`` 运行在上下文的副本里。

    因此 ``_session_value`` 里的 ``var.set()`` 在工具接口路径下不会回流到调用方，
    下一次未绑定的调用会重新随机。这不是缺陷 —— 生产路径永远先经
    ``apply_session_context`` 绑定，工具对上下文**只读**；这条测试是防止后人
    误以为兜底值会被固化而写出依赖它的代码。
    """
    agent_tools._session_user_id.set(None)

    get_user_id.invoke({})

    assert agent_tools._session_user_id.get() is None, (
        "工具调用内的写入不应回流 —— 若这条断言开始失败，说明机制变了，"
        "需要重新审视 _session_value 的兜底语义"
    )


# ----------------------------------------------------------------------
# 兜底
# ----------------------------------------------------------------------

def test_unbound_tool_returns_a_valid_reader_without_raising(bound_reader):
    """脱离会话上下文时不该抛异常，而是兜底给一位种子读者。

    注意断言的是「不抛异常且值合法」，不是「值固定」—— 理由见上一条。
    """
    agent_tools._session_user_id.set(None)
    agent_tools._session_month.set(None)

    assert get_user_id.invoke({}) in {f"{1000 + i}" for i in range(1, 11)}
    assert get_current_month.invoke({}) in agent_tools.month_arr


def test_direct_function_call_fallback_is_stable(bound_reader):
    """直接调用函数（不经工具接口）时，兜底值会固化下来。

    这条覆盖的是「脚本里手工 import 函数来调试」的用法。
    """
    agent_tools._session_user_id.set(None)

    first = agent_tools._session_value(agent_tools._session_user_id, ["1001", "1002"])

    assert first in {"1001", "1002"}
    assert agent_tools._session_value(agent_tools._session_user_id, ["1001", "1002"]) == first
