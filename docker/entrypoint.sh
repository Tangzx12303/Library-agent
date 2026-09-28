#!/bin/sh
# ==============================================================
# 容器启动引导：等 MySQL 就绪 → 按需建库建表灌种子 → 交给 CMD 启动 Streamlit
#
# 为什么要等：compose 的 depends_on + healthcheck 已能保证 MySQL 先就绪，
# 但单独用 docker run 起容器时没有这层保护。「首次启动自动初始化」这件事
# 必须在应用接收请求前做完，否则底层的建表语句会一路抛异常。
# ==============================================================
set -e

DB_HOST="${MYSQL_HOST:-127.0.0.1}"
DB_PORT="${MYSQL_PORT:-3306}"
DB_WAIT_SECONDS="${DB_WAIT_SECONDS:-90}"

echo "[entrypoint] 等待 MySQL ${DB_HOST}:${DB_PORT} 就绪（最多 ${DB_WAIT_SECONDS}s）..."
waited=0
until python -c "
import os, socket, sys
try:
    socket.create_connection((os.environ.get('MYSQL_HOST', '127.0.0.1'),
                              int(os.environ.get('MYSQL_PORT', '3306'))), timeout=2).close()
except OSError:
    sys.exit(1)
"; do
    waited=$((waited + 2))
    if [ "$waited" -ge "$DB_WAIT_SECONDS" ]; then
        echo "[entrypoint] 等待 MySQL 超时；仍继续启动应用（数据库不可用时前端会报错）" >&2
        break
    fi
    sleep 2
done

# db.init_db 是幂等的（IF NOT EXISTS / INSERT IGNORE，借阅明细仅空表时灌入），
# 因此每次启动都跑一遍是安全的，且不会冲掉管理员在后台做的改动。
if [ "${INIT_DB_ON_START:-1}" = "1" ]; then
    echo "[entrypoint] 初始化数据库（幂等）..."
    python -m db.init_db || echo "[entrypoint] 数据库初始化失败，继续启动应用" >&2
fi

echo "[entrypoint] 启动：$*"
exec "$@"
