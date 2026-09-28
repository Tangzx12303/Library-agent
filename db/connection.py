"""MySQL 连接与查询封装。

连接策略：**每次操作新建连接、用完即关**，不做连接池。
原因是 Streamlit 每次交互都会重跑整个脚本，且执行线程可能变化；
把 pymysql 连接缓存在模块级（或 st.cache_resource）里容易出现
「连接已被服务端断开 / 跨线程使用」的偶发报错。演示场景并发极低，
逐次建连的开销可以接受，换来的是行为确定。
"""
import os

import pymysql
from pymysql.cursors import DictCursor

from utils.config_handler import db_conf
from utils.logger_handler import logger


def get_connection(db: str | None = None) -> pymysql.connections.Connection:
    """建立一个新的 MySQL 连接。

    :param db: 指定要连接的库；默认用 config/db.yml 里的 database。
               建库脚本在库还不存在时需要传 None 之外的空串以跳过选库，
               故这里允许显式传入。
    """
    password = os.environ.get("MYSQL_PASSWORD")
    if password is None:
        raise RuntimeError(
            "未设置 MYSQL_PASSWORD 环境变量，请在 .env 中配置 MySQL 口令"
        )

    return pymysql.connect(
        host=db_conf["host"],
        port=int(db_conf["port"]),
        user=db_conf["user"],
        password=password,
        database=db_conf["database"] if db is None else db,
        charset=db_conf.get("charset", "utf8mb4"),
        cursorclass=DictCursor,
        autocommit=True,
    )


def get_server_connection() -> pymysql.connections.Connection:
    """不选库的连接——建库脚本在目标库尚不存在时需要它。"""
    return get_connection(db="")


def query_all(sql: str, params: tuple | list | None = None) -> list[dict]:
    """执行查询，返回全部行（每行是 dict）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()
    finally:
        conn.close()


def query_one(sql: str, params: tuple | list | None = None) -> dict | None:
    """执行查询，返回第一行；无结果返回 None。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()
    finally:
        conn.close()


def execute(sql: str, params: tuple | list | None = None) -> int:
    """执行写操作，返回受影响行数。

    连接开了 autocommit，无需显式 commit。
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            rows = cur.execute(sql, params or ())
            return rows
    except Exception as e:
        logger.error(f"[db]执行失败：{sql[:120]} | 参数={params} | 原因={str(e)}")
        raise
    finally:
        conn.close()
