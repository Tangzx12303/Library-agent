from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk

from utils.config_handler import agent_conf
from utils.prompt_loader import load_system_prompts
from agent.memory import build_messages
from agent.tools.agent_tools import (make_rag_summarize_tool, search_book,
                                     search_books_by_category,
                                     check_seat, get_opening_hours, get_user_id,
                                     get_current_month, fetch_external_data,
                                     fill_context_for_report, apply_session_context)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

# 未在 config/agent.yml 中配置时的默认值
# 历史消息的 token 预算：主阈值，按 token 而非条数控制上下文规模
DEFAULT_MAX_HISTORY_TOKENS = 2000
# 条数安全网：防止大量极短消息把条数撑到几百条
DEFAULT_MAX_HISTORY_MESSAGES = 50

# create_agent 构建的图中，顶层模型节点的名称。
# stream_mode="messages" 会抛出图内**所有** LLM 的输出，需要靠节点名区分哪些是
# 面向读者的正文（model 节点），哪些是工具内部调用（tools 节点）。
MODEL_NODE_NAME = "model"


class ReactAgent:
    def __init__(self, model):
        """:param model: 对话模型实例（由 app 层按用户选中的模型构造并缓存）。
                        它同时也是 RAG 总结与记忆压缩所用的模型，保证一次对话内
                        主回答、检索总结、历史压缩三条链路用的是同一个模型。
        """
        self.model = model
        self.agent = create_agent(
            model=model,
            system_prompt=load_system_prompts(),
            tools=[make_rag_summarize_tool(model), search_book, search_books_by_category,
                   check_seat, get_opening_hours, get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        self.max_history_tokens = agent_conf.get(
            "max_history_tokens", DEFAULT_MAX_HISTORY_TOKENS
        )
        self.max_history_messages = agent_conf.get(
            "max_history_messages", DEFAULT_MAX_HISTORY_MESSAGES
        )
        self.memory_summary_enabled = agent_conf.get("memory_summary_enabled", True)

    def prepare(self, query: str, history: list[dict] | None = None,
                summary: str = "") -> dict:
        """组装本轮输入：按 token 预算裁剪历史，把溢出部分压缩进记忆摘要。

        与 execute_stream 拆成两步，是因为「压缩」可能触发一次额外的模型调用
        （同步、非流式），而流式接口只负责产出正文。调用方先 prepare 拿到结果、
        把更新后的摘要落库，再把 messages 交给 execute_stream。

        :return: ``{"messages": [...], "summary": str, "changed": bool}``，
                 changed 为 True 时调用方应把 summary 写回持久化存储。
        """
        return build_messages(
            query=query,
            history=history,
            summary=summary,
            max_tokens=self.max_history_tokens,
            max_messages=self.max_history_messages,
            enabled=self.memory_summary_enabled,
            model=self.model,
        )

    def execute_stream(self, messages: list[dict],
                       session_context: dict[str, str] | None = None,
                       memory_summary: str = "",
                       trace: list[dict] | None = None):
        """以 token 粒度流式输出 Agent 的思考与最终回答，降低首字延迟。

        :param messages: 已组装好的完整输入消息列表（由 prepare 产出），
                         格式 ``[{"role": "user"/"assistant", "content": str}, ...]``，
                         末条即本轮提问。裁剪与压缩都已在 prepare 中完成。
        :param session_context: 本次对话固定的上下文（读者ID、月份），
                        由调用方持有并每轮传入，保证整个对话内身份一致。
        :param memory_summary: 溢出历史的压缩摘要，经 dynamic_prompt 中间件
                        注入 System Prompt；为空表示暂无记忆。
        :param trace: 调用方持有的空列表；本轮的工具调用会被中间件逐个追加上去，
                        供质检使用。传 None 表示不采集（不传即静默跳过）。
        """
        # 注入本次对话的固定上下文（读者ID / 月份）
        apply_session_context(session_context)

        for message_chunk, metadata in self.agent.stream(
            {"messages": messages},
            stream_mode="messages",
            # 这个 dict 有意保持「无 schema」：
            # create_agent 的 context_schema 为 None 时，LangGraph 会把它**原样**
            # 放进 Runtime.context（既不拷贝也不做类型转换），因此同一次 stream()
            # 内所有中间件共享同一个 dict 对象 —— "report" 标记与 "tool_calls"
            # 轨迹正是靠这一点跨节点互相可见的。
            # ⚠️ 若将来给 create_agent 传入 context_schema，_coerce_context 会改成
            #    context_schema(**context)，这两个键会一起失效。
            context={"report": False, "memory_summary": memory_summary,
                     "tool_calls": trace},
        ):
            # stream_mode="messages" 会把图内所有 LLM 输出都抛出来，其中混有两类噪音：
            #   1. tools 节点的 ToolMessage —— 即工具返回值（读者ID、借阅记录JSON、检索原文）；
            #   2. 工具内部嵌套调用的 LLM 输出 —— 例如 rag_summarize 内部的总结链，
            #      它同样是 AIMessageChunk，但属于工具内部实现，不该展示给读者。
            # 两者都只能靠节点名过滤掉，只保留顶层模型节点产出的、面向读者的正文。
            if metadata.get("langgraph_node") != MODEL_NODE_NAME:
                continue

            if not isinstance(message_chunk, AIMessageChunk):
                continue

            content = message_chunk.content
            if isinstance(content, str):
                if content:
                    yield content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield block.get("text", "")
