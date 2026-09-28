import contextvars
import json
import random

from langchain_core.tools import tool

from db import repository as repo
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger

# ----------------------------------------------------------------------
# RAG 工具：按模型构造 + 惰性初始化
# ----------------------------------------------------------------------
# RagSummarizeService 会连接 Chroma 向量库，属于重型资源。
# 若在模块导入期构造，则任何 `import agent_tools` 都会附带建立向量库连接，
# 因此改为首次调用工具时再构造。
#
# 又因为 RAG 总结要与主对话用同一个模型（切换模型时二者同步），工具不能是模块级
# 单例，改为「每个模型一份」的工厂：服务的惰性状态放在闭包里，与工具实例同生命周期。
def make_rag_summarize_tool(model):
    """:param model: 当前选中的对话模型实例，透传给 RagSummarizeService。"""
    rag_service: RagSummarizeService | None = None

    @tool(description="从向量存储中检索图书馆相关资料")
    def rag_summarize(query: str) -> str:
        nonlocal rag_service
        if rag_service is None:
            rag_service = RagSummarizeService(model)
        return rag_service.rag_summarize(query)

    return rag_summarize


# ----------------------------------------------------------------------
# 会话上下文（整个对话内固定）
# ----------------------------------------------------------------------
# get_user_id / get_current_month 原本每次调用都重新随机取值，会导致：
#   1. 同一次提问中 LLM 若调用两次 get_user_id，可能拿到两个不同读者ID，报告数据自相矛盾；
#   2. 多轮对话中追问「再推荐几本」时读者会换人，上下文不连贯；
#   3. 同一问题重复提问结果完全不同，无法复现。
# 现改为「一次对话固定一份上下文」：
#   - 读者登录后由 app 层用其真实 reader_id 构造上下文，存入 st.session_state；
#   - 此后每轮提问都把这同一份上下文传入 execute_stream()，整个对话内读者ID/月份不变；
#   - 「清空对话」时换一个月份重新开始，但**读者身份不变**（人还是同一个人）。
# 工具层用 contextvars 接收（而非直接 import streamlit），保证工具函数与前端框架解耦。
_session_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_user_id", default=None
)
_session_month: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_month", default=None
)

month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]


def new_session_context(user_id: str | None = None) -> dict[str, str]:
    """构造一份会话上下文（读者ID + 月份）。

    读者身份在登录时确定（传入 user_id），月份随机取一个已有借阅数据的月份。
    user_id 为空时随机取一位——仅用于脱离登录页单独调用 Agent 的兜底场景。
    """
    if user_id is None:
        user_id = random.choice([f"{1000 + i}" for i in range(1, 11)])
    return {"user_id": user_id, "month": random.choice(month_arr)}


def apply_session_context(session_context: dict[str, str] | None = None) -> dict[str, str]:
    """将会话上下文注入当前执行上下文，供工具函数读取。

    由 ReactAgent.execute_stream 在每轮提问前调用。每轮都用调用方持有的
    同一份上下文重新注入，因此不依赖线程/协程是否被复用——即便 Streamlit
    换了执行线程，值也不会丢。未传入时自动生成一份（便于脱离前端单独调用）。
    """
    context = dict(session_context) if session_context else new_session_context()

    _session_user_id.set(context.get("user_id"))
    _session_month.set(context.get("month"))

    logger.debug(
        f"[会话上下文]本轮使用 user_id={_session_user_id.get()}，month={_session_month.get()}"
    )
    return context


def _session_value(var: contextvars.ContextVar, candidates: list[str]) -> str:
    """取会话上下文中的值；若尚未注入则先注入再返回（脱离 Agent 直接调用时的兜底）。"""
    value = var.get()
    if value is None:
        value = random.choice(candidates)
        var.set(value)
    return value


def _authorize_reader(user_id: str) -> str | None:
    """校验借阅查询的目标读者是否为当前登录者。

    ``fetch_external_data`` 的 user_id 入参由 LLM 填写，因此必须在这里兜一道：
    模型的输出不是可信输入。

    **这里直接读 ``_session_user_id.get()``，不能走 ``_session_value``。**
    后者在值缺失时会随机兜底一个 ID —— 那会把一道鉴权变成抛硬币：未绑定身份
    的调用反而可能"恰好匹配"。

    :return: 通过返回 None；否则返回可直接回灌给模型的拒绝原因。
    """
    current = _session_user_id.get()
    if current is None:
        # 脱离会话上下文直接调用（脚本调试）。此时没有可信身份可比对，拒绝。
        logger.warning(f"[越权拦截]会话未绑定读者身份，拒绝查询 reader_id={user_id} 的借阅记录")
        return "当前会话未绑定读者身份，出于隐私保护，无法查询借阅记录"

    if str(user_id).strip() != str(current).strip():
        logger.warning(
            f"[越权拦截]请求查询 reader_id={user_id}，当前登录 reader_id={current}，已拒绝"
        )
        return f"出于隐私保护，只能查询当前登录读者（ID：{current}）本人的借阅记录"

    return None


# ----------------------------------------------------------------------
# 工具集
# ----------------------------------------------------------------------
@tool(description="根据书名查询图书馆藏情况，返回作者、类别、索书号、馆藏位置、"
                  "可借数量与总复本数，以纯字符串形式返回")
def search_book(book_name: str) -> str:
    books = repo.search_book_by_title(book_name)
    if not books:
        return f"《{book_name}》暂未检索到馆藏，建议通过馆际互借、文献传递或资源荐购获取"

    lines = []
    for b in books:
        lines.append(
            f"《{b['title']}》作者：{b['author']}，类别：{b['category']}，"
            f"索书号：{b['call_number']}，馆藏位置：{b['location']}，"
            f"可借 {b['available_stock']} 本 / 共 {b['total_stock']} 本，状态：{b['status']}"
        )
    return "\n".join(lines)


@tool(description="根据图书类别查询该类别下**当前可借（可借数量大于0）**的图书清单，"
                  "返回书名、作者、索书号、馆藏位置与可借数量。"
                  "生成图书推荐前必须调用本工具获取真实在架书目，不得凭常识推荐")
def search_books_by_category(category: str) -> str:
    books = repo.search_books_by_category(category, only_available=True)
    if not books:
        return f"类别「{category}」下暂无在架可借的图书"

    lines = [f"类别「{category}」下当前在架可借图书共 {len(books)} 种："]
    for i, b in enumerate(books, 1):
        lines.append(
            f"{i}. 《{b['title']}》作者：{b['author']}，索书号：{b['call_number']}，"
            f"馆藏位置：{b['location']}，可借 {b['available_stock']} 本"
        )
    return "\n".join(lines)


@tool(description="查询指定区域的自习/阅览座位剩余情况，以纯字符串形式返回")
def check_seat(area: str) -> str:
    remain = random.randint(0, 60)
    return f"{area}当前剩余座位约{remain}个，高峰期建议提前预约"


@tool(description="查询图书馆各功能分区的开放时间，以纯字符串形式返回")
def get_opening_hours() -> str:
    return ("图书馆开放时间：周一至周日 09:00-21:00；"
            "借阅区/阅览区 09:00-21:00；自习区 08:30-22:00；"
            "少儿阅览区 09:00-18:00（周二闭馆）；"
            "电子阅览室 09:00-20:30；法定节假日开放时间以馆内公告为准")


@tool(description="获取当前登录读者的读者ID，以纯字符串形式返回。整个对话内该ID保持不变，"
                  "若上文已经获取过读者ID则无需重复调用")
def get_user_id() -> str:
    return _session_value(_session_user_id, [f"{1000 + i}" for i in range(1, 11)])


@tool(description="获取当前月份，以纯字符串形式返回。整个对话内该月份保持不变，"
                  "若上文已经获取过月份则无需重复调用")
def get_current_month() -> str:
    return _session_value(_session_month, month_arr)


@tool(description="从数据库中获取指定读者在指定月份的图书借阅记录，以纯字符串形式返回，"
                  "如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    # 入参由 LLM 生成，先规范化再使用。
    # 必须**先** strip 再拿去查库：MySQL 的 = 比较对前导空格敏感，
    # 传 "  1005  " 会判不等于 "1005" 而静默查空，读者看到「未查到借阅记录」。
    user_id = str(user_id or "").strip()
    month = str(month or "").strip()

    # 越权防线：user_id 由 LLM 填写，必须校验它是当前登录读者本人。
    # 返回拒绝话术而非抛异常——抛异常会中断 ReAct 循环，读者看到的是一坨报错；
    # 返回话术则让模型能自然地把「无权查询他人」转述给读者。
    denial = _authorize_reader(user_id)
    if denial is not None:
        return denial

    row = repo.get_borrow_monthly(user_id, month)
    if row is None:
        logger.warning(f"[fetch_external_data]未能检索到读者：{user_id}在{month}的借阅记录")
        return ""

    # 出参结构与原 CSV 版本保持一致，报告提示词无需改动
    record = {
        "偏好类别": row["preference"],
        "月度借阅量": f"月借{row['borrow_count']}册 | 阅读时长{row['reading_hours']}h",
        "在借图书": row["current_books"],
        "对比分析": row["comparison"],
    }
    return json.dumps(record, ensure_ascii=False)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
