"""借阅查询的越权防线。

``fetch_external_data`` 的 user_id 入参由 LLM 填写——模型的输出不是可信输入。
这些用例守的是「读者 A 不能通过诱导模型查到读者 B 的借阅数据」。
"""
import pytest

from agent.tools import agent_tools
from agent.tools.agent_tools import fetch_external_data


@pytest.fixture
def borrow_stub(monkeypatch):
    """把仓库层换成记录调用的替身，返回一笔固定的借阅记录。"""
    calls: list[tuple] = []

    def fake_get_borrow_monthly(reader_id, month):
        calls.append((reader_id, month))
        return {"preference": "社科读者 | 历史", "borrow_count": 4,
                "reading_hours": 21, "current_books": "《万历十五年》",
                "comparison": "较上月持平"}

    monkeypatch.setattr(agent_tools.repo, "get_borrow_monthly", fake_get_borrow_monthly)
    return calls


# ----------------------------------------------------------------------
# 拒绝路径：这些都必须在**触碰仓库层之前**就返回
# ----------------------------------------------------------------------

def test_other_reader_is_denied_and_db_never_touched(bound_reader, borrow_stub):
    """登录 1001 却查 1002 —— 拒绝，且仓库层一次都不能被调用。"""
    bound_reader("1001")

    result = fetch_external_data.invoke({"user_id": "1002", "month": "2025-04"})

    assert "只能查询当前登录读者" in result
    assert borrow_stub == [], "越权请求绝不允许落到数据库层"


def test_unbound_session_is_denied(bound_reader, borrow_stub):
    """会话未绑定身份时必须拒绝。

    这里守的是一个很容易写错的地方：如果鉴权走 ``_session_value``，它在值缺失时
    会**随机兜底**一个读者ID，于是未绑定身份的请求反而可能"恰好匹配"而放行——
    一道鉴权就变成了抛硬币。
    """
    agent_tools._session_user_id.set(None)

    result = fetch_external_data.invoke({"user_id": "1005", "month": "2025-04"})

    assert "未绑定读者身份" in result
    assert borrow_stub == []


# ----------------------------------------------------------------------
# 放行路径
# ----------------------------------------------------------------------

def test_own_reader_id_passes_through(bound_reader, borrow_stub):
    """查询本人借阅记录正常放行，并且真的查了库。"""
    bound_reader("1005")

    result = fetch_external_data.invoke({"user_id": "1005", "month": "2025-04"})

    assert borrow_stub == [("1005", "2025-04")]
    assert "偏好类别" in result


def test_whitespace_and_int_like_ids_are_normalized(bound_reader, borrow_stub):
    """模型偶尔会给 ID 包上空白 —— 比较前要 strip，否则会误拒正常请求。"""
    bound_reader("1005")

    fetch_external_data.invoke({"user_id": "  1005  ", "month": "2025-04"})

    assert borrow_stub == [("1005", "2025-04")]


def test_authorize_reader_returns_none_on_match(bound_reader):
    """``_authorize_reader`` 的契约：通过返回 None，拒绝返回可读原因。"""
    bound_reader("1001")

    assert agent_tools._authorize_reader("1001") is None
    assert agent_tools._authorize_reader("1002") is not None


def test_authorize_reader_does_not_use_random_fallback(bound_reader, monkeypatch):
    """鉴权路径不得触碰 ``_session_value``（它会随机兜底）。"""
    agent_tools._session_user_id.set(None)

    def _boom(*args, **kwargs):
        raise AssertionError("鉴权不得走 _session_value —— 它会随机兜底一个 ID")

    monkeypatch.setattr(agent_tools, "_session_value", _boom)

    assert agent_tools._authorize_reader("1005") is not None
