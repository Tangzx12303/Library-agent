# syntax=docker/dockerfile:1
# ==============================================================
# 智阅通 · 图书馆智能客服 —— 应用镜像
#
# 构建：docker build -t library-agent:latest .
# 运行：推荐用 docker-compose.yml（会一并拉起 MySQL）
# 说明：完整部署步骤见 DOCKER_DEPLOY.md
#
# 基础镜像选 3.13-slim：与本地 .venv（Python 3.13.15）保持一致，
# 项目用到 match / 海象运算符，要求 >= 3.10。
# ==============================================================
FROM python:3.13-slim

# 国内网络下走清华镜像可显著加速。不在国内、或想用官方源时：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple \
#                --build-arg PIP_TRUSTED_HOST=pypi.org -t library-agent:latest .
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 运行时状态（向量库 + MD5 台账）统一收敛到 /app/state，这样只需挂载
# 一个数据卷就能同时持久化两者。二者必须同进同退——只留 chroma 而丢 md5，
# load_document() 会认为文件都已入库，得到「看起来正常但永远是旧的」向量库。
ENV CHROMA_PERSIST_DIR=/app/state/chroma \
    CHROMA_MD5_STORE=/app/state/md5.txt

# tzdata 仅为让日志时间戳与宿主机一致（slim 镜像默认没有时区库，只设 TZ 会退回 UTC）
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖再拷代码：requirements.txt 没变时这层直接命中缓存，改代码不会重装依赖
COPY requirements.txt ./
RUN pip install --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        --timeout 120 -r requirements.txt

COPY . .

# 预置知识库：把仓库里已经构建好的 chroma_db / md5.txt 落到 /app/state。
# 首次以空数据卷启动时，Docker 会用镜像中该目录的内容填充卷，
# 因此开箱即有可检索的知识库，**无需重跑 load_document()**——
# 重新向量化会消耗 DashScope 的 embedding 额度。
# （entrypoint.sh 的换行符修正：在 Windows 上编辑过的脚本可能带 CRLF，
#  会让 /bin/sh 报 "no such file or directory"，这里统一切成 LF。）
RUN mkdir -p /app/state/chroma /app/logs \
 && if [ -d chroma_db ]; then cp -a chroma_db/. /app/state/chroma/; fi \
 && if [ -f md5.txt ]; then cp -a md5.txt /app/state/md5.txt; fi \
 && sed -i 's/\r$//' docker/entrypoint.sh \
 && chmod +x docker/entrypoint.sh

# 非 root 运行：应用本身不需要写 /app 之外的内容，被攻破时影响面更小
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# 用 python 自身探活，省掉在镜像里装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=4).status==200 else 1)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["streamlit", "run", "app.py"]
