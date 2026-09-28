from typing import Callable

from quality.trace import record_call, finish_call, extract_result_text
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    # 质检用的调用轨迹。context 里没有 "tool_calls" 键（或 context 不是 dict）时
    # record_call 收到 None，静默跳过——不采集轨迹不影响对话本身。
    context = getattr(request.runtime, "context", None)
    trace = context.get("tool_calls") if isinstance(context, dict) else None

    # 先登记再执行：工具抛异常时条目已经在了，回填 error 即可。
    entry = record_call(trace, request.tool_call)

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        finish_call(entry, result=extract_result_text(result))

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        finish_call(entry, error=f"{type(e).__name__}: {e}")
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        prompt = load_report_prompts()
    else:
        prompt = load_system_prompts()

    # 注入溢出历史的压缩摘要（由 ReactAgent.execute_stream 经 context 传入）。
    # 放在 System Prompt 而不是消息列表里：整个会话只该有一条 system 消息，
    # 塞进消息列表会与其 role 语义冲突，也会破坏消息的 user/assistant 交替结构。
    summary = request.runtime.context.get("memory_summary", "")
    if summary:
        prompt += f"\n\n### 之前的对话记忆\n{summary}"

    return prompt
