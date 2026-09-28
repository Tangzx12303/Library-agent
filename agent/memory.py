"""多轮对话记忆：token 预算控制 + 溢出压缩。

解决两件事：

1. **按 token 预算裁剪，而不是按条数**。条数不等于长度——一份 Markdown 推荐报告
   上千 token，一句「谢谢」十几个 token，两者都算 1 条，按条数截断时预算不可控。

2. **溢出的历史压缩而不是丢弃**。原来的做法是把超出窗口的消息直接扔掉，
   模型对此毫不知情；再叠加主提示词里「严禁反问读者、禁止仅因信息不完整就回
   『我不知道』」的约束，读者追问旧话题时模型只能靠猜。现在改为把溢出的部分
   交给模型合并进一份**滚动摘要**，摘要随 System Prompt 注入，并按读者ID持久化。

token 数用字符启发式估算（中文≈1字/token，英文≈4字符/token），不引入 tokenizer
依赖。它只需要是一个**稳定的预算刻度**，不承担计费职责，因此偏保守即可。
"""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from utils.logger_handler import logger
from utils.prompt_loader import load_memory_summary_prompts

# CJK 汉字 / 日文假名 / 中日韩与全角标点：中文模型的分词器对它们约等于 1 字符 = 1 token
_CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ　-〿＀-￯]")

# 把待压缩的历史拼成纯文本时使用的行分隔符
_TRANSCRIPT_SEPARATOR = "\n"

# 摘要链的输入键（与 prompts/memory_summary.txt 的占位符一致）
_EMPTY_SUMMARY_PLACEHOLDER = "（暂无）"


def estimate_tokens(text: str) -> int:
    """字符启发式估算 token 数。

    CJK 字符按 1:1 计，其余字符按 4:1 计（向上取整）。这不是精确分词，
    但它跨中英文稳定、零依赖、无网络调用——正好适合「给预算划刻度」这个用途。
    """
    if not text:
        return 0

    cjk_count = len(_CJK_PATTERN.findall(text))
    other_count = len(text) - cjk_count
    return cjk_count + (other_count + 3) // 4


def _clean_history(history: list[dict] | None) -> list[dict]:
    """只接受 user/assistant 的纯文本消息，过滤异常结构，避免脏数据污染上下文。"""
    if not history:
        return []

    cleaned: list[dict] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned


def _select_recent(history: list[dict], max_tokens: int,
                   max_messages: int) -> tuple[list[dict], list[dict]]:
    """从新到旧挑出能装进预算的历史，返回 ``(保留的, 溢出的)``。

    两个约束同时生效：token 预算是主阈值，条数是安全网。任一项设为 0 或负数
    表示不携带历史，此时全部历史都算溢出（开启压缩时会先被摘要掉，而非凭空消失）。
    """
    if not history:
        return [], []

    if max_tokens <= 0 or max_messages <= 0:
        return [], list(history)

    kept_reversed: list[dict] = []
    used_tokens = 0
    cut_index = 0

    for index in range(len(history) - 1, -1, -1):
        cost = estimate_tokens(history[index]["content"])

        # 最新一条无条件保留：它是最相关的上下文，若因单条超长而被压进摘要，
        # 「最近的对话」反而只剩一句二手概括，得不偿失。
        if kept_reversed and (len(kept_reversed) >= max_messages
                              or used_tokens + cost > max_tokens):
            cut_index = index + 1
            break

        kept_reversed.append(history[index])
        used_tokens += cost
    else:
        cut_index = 0

    return list(reversed(kept_reversed)), history[:cut_index]


def _summarize(overflow: list[dict], previous_summary: str, model) -> str:
    """把溢出的旧消息合并进已有摘要。

    任何失败都退回上一版摘要：压缩只是优化手段，不该让本轮问答因此中断
    （代价是这批旧消息本轮丢失，与「不开压缩」时的行为一致）。

    :param model: 当前选中的对话模型实例，与主对话共用同一个模型。
    """
    # 没有模型就无从压缩。正常路径不会走到这里（ReactAgent 必传），
    # 显式挡掉是为了避免拿 None 去拼链、在报错里留下一句误导性的「摘要生成失败」。
    if model is None:
        logger.warning("[对话记忆]未提供模型，跳过摘要压缩，退回上一版摘要")
        return previous_summary

    transcript = _TRANSCRIPT_SEPARATOR.join(
        f"{'读者' if item['role'] == 'user' else '客服'}：{item['content']}"
        for item in overflow
    )

    try:
        # 每次现取提示词文件内容（prompt_loader 有 mtime 缓存），
        # 因此改了 prompts/memory_summary.txt 无需重启即生效。
        chain = (
            PromptTemplate.from_template(load_memory_summary_prompts())
            | model
            | StrOutputParser()
        )
        merged = chain.invoke({
            "previous_summary": previous_summary or _EMPTY_SUMMARY_PLACEHOLDER,
            "new_messages": transcript,
        }).strip()

        logger.info(f"[对话记忆]已将 {len(overflow)} 条历史合并进摘要，"
                    f"摘要长度 {len(merged)} 字（约 {estimate_tokens(merged)} token）")
        return merged
    except Exception as e:
        logger.warning(f"[对话记忆]摘要生成失败，退回上一版摘要：{str(e)}")
        return previous_summary


def build_messages(query: str, history: list[dict] | None = None, summary: str = "",
                   max_tokens: int = 2000, max_messages: int = 50,
                   enabled: bool = True, model=None) -> dict:
    """组装本轮送入模型的消息列表，并维护滚动记忆摘要。

    :param query: 本轮读者提问（追加在历史之后）
    :param history: 之前的对话历史，不含本轮提问
    :param summary: 目前持有的记忆摘要（由调用方持久化并每轮传入）
    :param max_tokens: 历史消息的 token 预算
    :param max_messages: 历史消息条数上限（安全网）
    :param enabled: 是否启用压缩层；关闭则溢出部分直接丢弃
    :param model: 压缩摘要所用的对话模型实例（当前选中的模型）
    :return: ``{"messages": [...], "summary": str, "changed": bool}``。
             ``changed`` 为 True 表示摘要已更新，调用方应把它写回持久化存储。
    """
    cleaned = _clean_history(history)
    kept, overflow = _select_recent(cleaned, max_tokens, max_messages)

    new_summary = summary or ""
    changed = False

    if overflow and enabled:
        new_summary = _summarize(overflow, new_summary, model)
        # 摘要生成失败时会退回原值，此时 changed 仍为 False，不必重复落库
        changed = new_summary != (summary or "")

    messages = kept + [{"role": "user", "content": query}]

    logger.debug(
        f"[对话记忆]历史 {len(cleaned)} 条：保留 {len(kept)} 条"
        f"（约 {sum(estimate_tokens(m['content']) for m in kept)} token）、"
        f"{'压缩' if enabled else '丢弃'} {len(overflow)} 条，"
        f"本轮送入模型 {len(messages)} 条"
    )

    return {"messages": messages, "summary": new_summary, "changed": changed}
