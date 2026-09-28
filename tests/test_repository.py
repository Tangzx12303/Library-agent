"""数据访问层：SQL 参数化、类别拆词、「只返回可借图书」这条硬约束。

**mock 的目标是 ``db.repository`` 上的名字，不是 ``db.connection`` 上的。**
repository 用的是 ``from db.connection import query_all``，所以打
``db.connection.query_all`` 对它毫无效果 —— 那会让测试在「通过」的同时真的连上
数据库。这是本项目最容易踩的 mock 陷阱。
"""
import pytest

from db import repository as repo


@pytest.fixture
def captured(monkeypatch):
    """替换 query_all，记录下发的 SQL 与参数。"""
    calls: list[tuple[str, list]] = []

    def fake_query_all(sql, params=None):
        calls.append((sql, list(params or [])))
        return []

    monkeypatch.setattr(repo, "query_all", fake_query_all)
    return calls


# ----------------------------------------------------------------------
# search_books_by_category：类别拆词
# ----------------------------------------------------------------------

def test_reader_persona_string_is_split_into_keywords(captured):
    """读者的「偏好类别」是形如 ``文学爱好者 | 科幻/历史`` 的画像串。

    整串拿去 LIKE 一个类别都匹配不上，因此必须按分隔符拆词再 OR 匹配。
    """
    repo.search_books_by_category("文学爱好者 | 科幻/历史")

    sql, params = captured[0]
    assert sql.count("category LIKE %s") == 3
    assert " OR " in sql
    assert params == ["%文学爱好者%", "%科幻%", "%历史%"]


@pytest.mark.parametrize("raw,expected", [
    ("科幻/历史", ["%科幻%", "%历史%"]),
    ("科幻、历史", ["%科幻%", "%历史%"]),
    ("科幻,历史", ["%科幻%", "%历史%"]),
    ("科幻，历史", ["%科幻%", "%历史%"]),
    ("科幻 历史", ["%科幻%", "%历史%"]),
    ("  科幻  ", ["%科幻%"]),
    ("科幻", ["%科幻%"]),
])
def test_all_supported_separators(captured, raw, expected):
    repo.search_books_by_category(raw)

    assert captured[0][1] == expected


def test_blank_category_skips_the_query_entirely(captured):
    """空/全分隔符的类别不该白跑一次数据库。"""
    for blank in ("", "   ", "/|/", "、，"):
        assert repo.search_books_by_category(blank) == []

    assert captured == []


# ----------------------------------------------------------------------
# 「只返回可借图书」—— 本项目的核心业务约束
# ----------------------------------------------------------------------

def test_only_available_pushes_constraint_into_sql(captured):
    """约束必须出现在 WHERE 子句里，不能藏在 Python 层过滤。

    这是「模型能推荐的书目集合在数据层面就被限死」的实现位置 —— 它在 SQL 里，
    所以不依赖模型是否听话。
    """
    repo.search_books_by_category("科幻", only_available=True)

    sql, _ = captured[0]
    assert "available_stock > 0" in sql


def test_only_available_false_omits_the_constraint(captured):
    """关掉约束时（管理员视角）才允许看到已借空的书。"""
    repo.search_books_by_category("科幻", only_available=False)

    sql, _ = captured[0]
    assert "available_stock > 0" not in sql


def test_keyword_count_does_not_change_parameterization(captured):
    """动态拼的是占位符个数，不是值本身 —— 拆词不得引入拼接注入面。"""
    repo.search_books_by_category("科幻/历史/哲学")

    sql, params = captured[0]
    assert sql.count("%s") == 3
    assert len(params) == 3
    assert all(p.startswith("%") and p.endswith("%") for p in params)


# ----------------------------------------------------------------------
# 其它 DAO：确认走的都是参数化，而非字符串拼接
# ----------------------------------------------------------------------

def test_parameterized_writes(captured, monkeypatch):
    monkeypatch.setattr(repo, "execute", lambda sql, params=None: (captured.append((sql, list(params or []))), 1)[1])

    repo.update_user_status(7, "disabled")

    sql, params = captured[0]
    assert params == ["disabled", 7]
    assert "7" not in sql and "disabled" not in sql, "值必须走占位符"


def test_save_memory_summary_uses_upsert(captured, monkeypatch):
    """用 ON DUPLICATE KEY UPDATE 而非「先查再写」：省一次往返，且并发下
    不会出现两个请求都判定「不存在」而双双 INSERT 撞主键。"""
    monkeypatch.setattr(repo, "execute", lambda sql, params=None: (captured.append((sql, list(params or []))), 1)[1])

    repo.save_memory_summary("1005", "摘要")

    sql, params = captured[0]
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert params == ["1005", "摘要"]


def test_list_readers_with_memory_uses_left_join(captured):
    """必须 LEFT JOIN：从未生成过记忆的读者也要出现在列表里（summary 为 NULL），
    管理员才能看到「谁还没有记忆」，而不是只看到有记忆的那部分人。"""
    repo.list_readers_with_memory()

    sql, _ = captured[0]
    assert "LEFT JOIN" in sql
