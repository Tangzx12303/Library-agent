"""质检记录的落库与读取，以及「永不抛异常」的门面。

mock 打的是 ``db.repository`` 上的名字（repository 用
``from db.connection import query_all`` 导入，打 ``db.connection`` 无效）。
"""
import json

import pytest

from db import repository as repo
from quality.evaluator import evaluate_turn
from quality.store import (safe_count_records, safe_evaluate, safe_list_models,
                           safe_load_records, safe_record_failure)


@pytest.fixture
def writer(monkeypatch):
    """捕获 insert 的参数。"""
    captured: list[dict] = []
    monkeypatch.setattr(repo, "insert_eval_turn_record",
                        lambda record: (captured.append(record), 1)[1])
    return captured


@pytest.fixture
def reader(monkeypatch):
    """替换 query_all，返回预置行并记录 SQL。"""
    calls: list[tuple[str, list]] = []
    rows: list[dict] = []

    def fake_query_all(sql, params=None):
        calls.append((sql, list(params or [])))
        return rows

    monkeypatch.setattr(repo, "query_all", fake_query_all)
    return calls, rows


def _report(**kwargs):
    kwargs.setdefault("query", "根据我的借阅数据进行个性化推荐")
    kwargs.setdefault("response", "## 推荐书目\n《西方美术史》索书号：J110.9/0043，可借 3 本")
    kwargs.setdefault("trace", [])
    kwargs.setdefault("reader_id", "1005")
    kwargs.setdefault("model_id", "qwen3.8-max")
    return evaluate_turn(**kwargs)


# ----------------------------------------------------------------------
# 落库
# ----------------------------------------------------------------------

def test_to_row_columns_match_the_insert_statement(writer, monkeypatch):
    """``to_row()`` 的键必须与 INSERT 的列一一对应，否则会静默丢字段。"""
    monkeypatch.setattr(repo, "execute", lambda sql, params=None: 1)
    captured = []
    monkeypatch.setattr(repo, "insert_eval_turn_record",
                        lambda record: captured.append(record) or 1)

    row = _report().to_row()
    missing = [column for column in repo._EVAL_COLUMNS if column not in row]

    assert missing == []
    assert set(row) >= set(repo._EVAL_COLUMNS)


def test_insert_uses_parameterized_placeholders(monkeypatch):
    captured = []

    def fake_execute(sql, params=None):
        captured.append((sql, list(params or [])))
        return 1

    monkeypatch.setattr(repo, "execute", fake_execute)

    repo.insert_eval_turn_record({c: None for c in repo._EVAL_COLUMNS})

    sql, params = captured[0]
    assert sql.count("%s") == len(repo._EVAL_COLUMNS)
    assert len(params) == len(repo._EVAL_COLUMNS)
    assert "qwen3.8-max" not in sql, "值必须走占位符"


def test_insert_missing_keys_become_null(monkeypatch):
    """缺键按 NULL 处理，不抛 KeyError —— 崩溃轮的记录只有部分字段。"""
    captured = []
    monkeypatch.setattr(repo, "execute",
                        lambda sql, params=None: captured.append(params) or 1)

    repo.insert_eval_turn_record({"reader_id": "1005", "model_id": "m"})

    params = captured[0]
    assert params[0] == "1005"
    assert params[6] is None            # passed 缺席 → NULL


# ----------------------------------------------------------------------
# 不适用必须是 NULL，不是 0
# ----------------------------------------------------------------------

def test_not_applicable_metrics_are_stored_as_none(writer):
    """纯知识问答没有推荐 —— 比率列必须落 NULL。

    若落 0，每个知识类问题都会把管理员的推荐 F1 均值往下拽；而 MySQL 的 AVG()
    会自动跳过 NULL，问答分布变化时趋势图才不会出现假塌陷。
    """
    safe_evaluate(query="图书馆有哪些功能分区？", response="图书馆分为借阅区…",
                  trace=[], reader_id="1005", model_id="m")

    row = writer[0]
    assert row["rec_precision"] is None
    assert row["rec_recall"] is None
    assert row["rec_f1"] is None
    assert row["determined"] >= 1


def test_query_is_truncated_before_insert(writer):
    """超长问题会在 INSERT 阶段失败，先截断。"""
    safe_evaluate(query="问" * 900, response="答", trace=[],
                  reader_id="1005", model_id="m")

    assert len(writer[0]["query"]) == 512


def test_detail_json_excludes_raw_tool_output(writer):
    result = "类别「绘画」下当前在架可借图书共 1 种：\n1. 《西方美术史》可借 3 本"
    trace = [{"seq": 0, "name": "search_books_by_category", "args": {},
              "result": result, "error": None}]

    safe_evaluate(query="推荐几本绘画的书", response="推荐《西方美术史》，索书号：J110.9/0043",
                  trace=trace, reader_id="1005", model_id="m")

    detail = writer[0]["detail_json"]
    assert "可借 3 本" not in detail          # 工具原文不进库
    assert "西方美术史" in detail              # 派生事实要留


# ----------------------------------------------------------------------
# 失败隔离
# ----------------------------------------------------------------------

def test_compute_failure_returns_none_instead_of_raising(monkeypatch):
    """指标计算炸了也不能影响对话。"""
    import quality.store as store

    monkeypatch.setattr(store, "evaluate_turn",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("炸了")))

    assert safe_evaluate(query="问", response="答", trace=[], reader_id="1") is None


def test_persist_failure_still_returns_the_report(monkeypatch):
    """落库失败不该让读者看不到质检结果 —— 存不存得下与看不看得到是两回事。"""
    monkeypatch.setattr(repo, "insert_eval_turn_record",
                        lambda record: (_ for _ in ()).throw(RuntimeError("库挂了")))

    report = safe_evaluate(query="你好", response="您好", trace=[],
                           reader_id="1005", model_id="m")

    assert report is not None
    assert report.passed_count >= 1


def test_load_records_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(repo, "list_eval_records",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("库挂了")))

    assert safe_load_records() == []


def test_count_returns_zero_on_failure(monkeypatch):
    monkeypatch.setattr(repo, "count_eval_records",
                        lambda since=None: (_ for _ in ()).throw(RuntimeError("库挂了")))

    assert safe_count_records() == 0


def test_list_models_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(repo, "list_eval_models",
                        lambda: (_ for _ in ()).throw(RuntimeError("库挂了")))

    assert safe_list_models() == []


# ----------------------------------------------------------------------
# 崩溃轮次必须留痕
# ----------------------------------------------------------------------

def test_failed_turn_is_recorded(writer):
    """崩溃轮次必须留一行，否则看板有幸存者偏差。

    工具抛异常会中断整轮执行，response 为 None —— 如果不补记录，那一轮就从表里
    彻底消失，结果是**数据库越坏，看板看起来越好**。
    """
    safe_record_failure(query="推荐几本书", error="OperationalError: 连接失败",
                        reader_id="1005", model_id="m", latency_ms=800)

    row = writer[0]
    assert row["scenario"] == "failed"
    assert row["passed"] == 0
    assert row["determined"] == 1
    assert row["latency_ms"] == 800

    detail = json.loads(row["detail_json"])
    assert detail["checks"][0]["state"] == "fail"
    assert "连接失败" in detail["error"]
    # 崩溃轮不参与任何比率统计
    assert row["rec_f1"] is None and row["tool_recall"] is None
    assert row["tool_precision"] is None


def test_failed_turn_recording_never_raises(monkeypatch):
    monkeypatch.setattr(repo, "insert_eval_turn_record",
                        lambda record: (_ for _ in ()).throw(RuntimeError("库挂了")))

    safe_record_failure(query="问", error="炸了", reader_id="1005")   # 不应抛异常


# ----------------------------------------------------------------------
# 读取与过滤
# ----------------------------------------------------------------------

def test_list_records_applies_filters(monkeypatch):
    captured = []
    monkeypatch.setattr(repo, "query_all",
                        lambda sql, params=None: captured.append((sql, list(params or []))) or [])

    repo.list_eval_records(since="2026-09-01 00:00:00", limit=50,
                           model_ids=["qwen3.8-max", "glm-5.2"],
                           scenarios=["report"])

    sql, params = captured[0]
    assert "created_at >= %s" in sql
    assert sql.count("IN (%s, %s)") == 1        # 模型两个占位符
    assert "scenario IN (%s)" in sql
    assert "LIMIT %s" in sql
    assert params == ["2026-09-01 00:00:00", "qwen3.8-max", "glm-5.2", "report", 50]


def test_list_records_without_filters(monkeypatch):
    captured = []
    monkeypatch.setattr(repo, "query_all",
                        lambda sql, params=None: captured.append((sql, list(params or []))) or [])

    repo.list_eval_records()

    sql, params = captured[0]
    assert "created_at >=" not in sql
    assert " IN (" not in sql
    assert params == [2000]


def test_list_records_orders_newest_first(monkeypatch):
    captured = []
    monkeypatch.setattr(repo, "query_all",
                        lambda sql, params=None: captured.append((sql, list(params or []))) or [])

    repo.list_eval_records()

    assert "ORDER BY created_at DESC" in captured[0][0]


def test_count_records(monkeypatch):
    monkeypatch.setattr(repo, "query_one", lambda sql, params=None: {"c": 42})

    assert repo.count_eval_records() == 42
    assert repo.count_eval_records(since="2026-09-01") == 42


def test_load_records_parses_detail_json(monkeypatch):
    monkeypatch.setattr(repo, "list_eval_records", lambda **kwargs: [
        {"scenario": "report", "detail_json": json.dumps({"checks": []})},
        {"scenario": "other", "detail_json": "不是 JSON"},
        {"scenario": "other", "detail_json": None},
    ])

    rows = safe_load_records()

    assert rows[0]["detail"] == {"checks": []}
    assert rows[1]["detail"] == {}          # 脏数据降级为空字典，不抛
    assert rows[2]["detail"] == {}
