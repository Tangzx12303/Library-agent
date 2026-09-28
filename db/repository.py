"""数据访问层：把 SQL 收敛在一处，供 Agent 工具、登录页、管理后台共用。

返回值一律是 dict / list[dict]（连接已配 DictCursor），调用方无需再解析元组。
写操作遇到唯一键冲突等异常时直接向上抛，由调用方决定如何提示用户。
"""
import re

from db.connection import query_all, query_one, execute
from utils.security import hash_password

# ======================================================================
# 用户
# ======================================================================


def get_user_by_username(username: str) -> dict | None:
    return query_one("SELECT * FROM users WHERE username = %s", (username,))


def create_user(username: str, password: str, role: str = "reader",
                display_name: str | None = None,
                reader_id: str | None = None) -> int:
    """新建账号。reader_id 未指定时沿用 username（读者ID需唯一）。"""
    return execute(
        "INSERT INTO users (username, password_hash, role, display_name, reader_id) "
        "VALUES (%s, %s, %s, %s, %s)",
        (username, hash_password(password), role, display_name, reader_id or username),
    )


def list_users() -> list[dict]:
    return query_all(
        "SELECT id, username, role, display_name, reader_id, status, created_at "
        "FROM users ORDER BY role, reader_id"
    )


def list_readers() -> list[dict]:
    """只列读者，供管理员「借阅查询」下拉框使用。"""
    return query_all(
        "SELECT reader_id, username, display_name, status FROM users "
        "WHERE role = 'reader' ORDER BY reader_id"
    )


def update_user_status(user_id: int, status: str) -> int:
    return execute("UPDATE users SET status = %s WHERE id = %s", (status, user_id))


def reset_password(user_id: int, new_password: str) -> int:
    return execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password(new_password), user_id),
    )


def delete_user(user_id: int) -> int:
    return execute("DELETE FROM users WHERE id = %s", (user_id,))


# ======================================================================
# 图书
# ======================================================================


def search_book_by_title(title: str) -> list[dict]:
    """按书名模糊检索，返回所有匹配的书（含库存字段）。"""
    return query_all(
        "SELECT * FROM books WHERE title LIKE %s ORDER BY available_stock DESC, title",
        (f"%{title}%",),
    )


def search_books_by_category(category: str, only_available: bool = True) -> list[dict]:
    """按类别检索图书。

    入参常直接来自读者的「偏好类别」字段，形如 ``科幻/历史``、
    ``文学爱好者 | 科幻/历史``。因此这里按分隔符拆成多个关键词做 OR 匹配，
    而不是整串 LIKE——否则「文学爱好者 | 科幻/历史」一个类别都匹配不上。

    :param only_available: 为 True 时只返回「可借数量 > 0」的书——
                           推荐场景必须用这个模式，否则会推荐出借空的书。
    """
    keywords = [k.strip() for k in re.split(r"[/|、,，\s]+", category) if k.strip()]
    if not keywords:
        return []

    where = " OR ".join(["category LIKE %s"] * len(keywords))
    sql = f"SELECT * FROM books WHERE ({where})"
    params = [f"%{k}%" for k in keywords]

    if only_available:
        sql += " AND available_stock > 0"
    sql += " ORDER BY available_stock DESC, title"

    return query_all(sql, params)


def list_books() -> list[dict]:
    return query_all("SELECT * FROM books ORDER BY category, title")


def add_book(title: str, author: str, category: str, call_number: str,
             location: str, total_stock: int, available_stock: int | None = None) -> int:
    """新增图书。available_stock 未指定时默认等于 total_stock。"""
    if available_stock is None:
        available_stock = total_stock
    return execute(
        "INSERT INTO books (title, author, category, call_number, location, "
        "total_stock, available_stock, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (title, author, category, call_number, location, total_stock, available_stock,
         "在架可借" if available_stock > 0 else "已借空"),
    )


def update_book_stock(book_id: int, total_stock: int, available_stock: int) -> int:
    """调整库存，并同步维护状态文案，避免出现「可借 0 本却显示在架可借」。"""
    status = "在架可借" if available_stock > 0 else "已借空"
    return execute(
        "UPDATE books SET total_stock = %s, available_stock = %s, status = %s WHERE id = %s",
        (total_stock, available_stock, status, book_id),
    )


# ======================================================================
# 借阅
# ======================================================================


def get_borrow_monthly(reader_id: str, month: str) -> dict | None:
    """取某读者某月的借阅汇总，供推荐报告使用。"""
    return query_one(
        "SELECT * FROM borrow_monthly WHERE reader_id = %s AND month = %s",
        (reader_id, month),
    )


def get_borrow_monthly_all(reader_id: str) -> list[dict]:
    """取某读者的全部月度借阅汇总，按月份正序。"""
    return query_all(
        "SELECT month AS 月份, preference AS 偏好类别, borrow_count AS 月借册数, "
        "reading_hours AS 阅读时长h, current_books AS 在借图书, comparison AS 对比分析 "
        "FROM borrow_monthly WHERE reader_id = %s ORDER BY month",
        (reader_id,),
    )


def get_borrow_history(reader_id: str) -> list[dict]:
    """取某读者的借阅明细，按月份倒序，供管理员查询借阅历史。"""
    return query_all(
        "SELECT book_title, month, borrow_date, due_date, return_date, status "
        "FROM borrow_records WHERE reader_id = %s "
        "ORDER BY month DESC, book_title",
        (reader_id,),
    )


# ======================================================================
# 对话记忆
# ======================================================================


def get_memory_summary(reader_id: str) -> str | None:
    """取某读者的跨会话记忆摘要；从未产生过记忆时返回 None。

    由登录 / 清空对话时调用：读者重新进来就能续上上次对话的要点。
    """
    row = query_one(
        "SELECT summary FROM conversation_memory WHERE reader_id = %s",
        (reader_id,),
    )
    return row["summary"] if row else None


def save_memory_summary(reader_id: str, summary: str) -> int:
    """写入/更新某读者的记忆摘要。

    用 ON DUPLICATE KEY UPDATE 而非「先查再写」：一是省一次往返，
    二是并发下不会出现两个请求都判定「不存在」而双双 INSERT 撞主键。
    传入空串即为清除记忆（清空的是内容，不是行）。
    """
    return execute(
        "INSERT INTO conversation_memory (reader_id, summary) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE summary = VALUES(summary)",
        (reader_id, summary),
    )


def list_readers_with_memory() -> list[dict]:
    """列出所有读者及其长期记忆摘要，供管理员「记忆与摘要」视图使用。

    用 LEFT JOIN：即使某读者从未生成过记忆，也要出现在列表里（summary 为 None），
    让管理员能一目了然地看到「谁有记忆、谁还没有」，而非只看到有记忆的那部分人。
    """
    return query_all(
        "SELECT u.reader_id, u.username, u.display_name, u.status, "
        "       m.summary, m.updated_at "
        "FROM users u "
        "LEFT JOIN conversation_memory m ON m.reader_id = u.reader_id "
        "WHERE u.role = 'reader' "
        "ORDER BY u.reader_id"
    )


# ======================================================================
# 对话质检
# ======================================================================
# 记录由 quality.EvaluationReport.to_row() 产出，键名与此处一一对应。
# 比率列允许为 None —— 落库即 NULL，表示「本轮不适用」，MySQL 的 AVG() 会自动
# 跳过它们。**绝不能把不适用写成 0**：那会让知识问答轮次拉低推荐指标的均值。

_EVAL_COLUMNS = (
    "reader_id", "model_id", "query", "scenario", "scenario_source",
    "passed", "determined",
    "rec_precision", "rec_recall", "rec_f1",
    "tool_precision", "tool_recall", "tool_f1",
    "latency_ms", "detail_json",
)


def insert_eval_turn_record(record: dict) -> int:
    """写入一轮质检记录。``record`` 用 ``EvaluationReport.to_row()`` 的产出。"""
    columns = ", ".join(_EVAL_COLUMNS)
    placeholders = ", ".join(["%s"] * len(_EVAL_COLUMNS))
    return execute(
        f"INSERT INTO eval_turn_records ({columns}) VALUES ({placeholders})",
        tuple(record.get(column) for column in _EVAL_COLUMNS),
    )


def list_eval_records(since: str | None = None, limit: int = 2000,
                      model_ids: list[str] | None = None,
                      scenarios: list[str] | None = None) -> list[dict]:
    """按时间倒序取质检记录，供管理员看板聚合。

    :param since: 起始时间（``YYYY-MM-DD HH:MM:SS``），None 表示不限
    :param limit: 最多取多少行。看板是**聚合视图**，不需要把全表拉进内存
    :param model_ids: 只看这些模型；为空表示不限
    :param scenarios: 只看这些场景；为空表示不限
    """
    sql = "SELECT * FROM eval_turn_records WHERE 1 = 1"
    params: list = []

    if since:
        sql += " AND created_at >= %s"
        params.append(since)

    # 值仍走占位符，只是占位符的**个数**由列表长度决定
    if model_ids:
        sql += f" AND model_id IN ({', '.join(['%s'] * len(model_ids))})"
        params.extend(model_ids)

    if scenarios:
        sql += f" AND scenario IN ({', '.join(['%s'] * len(scenarios))})"
        params.extend(scenarios)

    sql += " ORDER BY created_at DESC, id DESC LIMIT %s"
    params.append(int(limit))

    return query_all(sql, params)


def count_eval_records(since: str | None = None) -> int:
    """质检记录总数（不受 list_eval_records 的 limit 影响）。"""
    if since:
        row = query_one(
            "SELECT COUNT(*) AS c FROM eval_turn_records WHERE created_at >= %s",
            (since,),
        )
    else:
        row = query_one("SELECT COUNT(*) AS c FROM eval_turn_records")
    return row["c"] if row else 0


def list_eval_models() -> list[str]:
    """看板筛选下拉框的模型选项 —— 只列出真正产生过记录的模型。"""
    rows = query_all(
        "SELECT DISTINCT model_id FROM eval_turn_records ORDER BY model_id"
    )
    return [row["model_id"] for row in rows]
