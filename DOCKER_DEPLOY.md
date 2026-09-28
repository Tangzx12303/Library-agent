# 智阅通 · 图书馆智能客服 —— Docker 部署指南

> 目标：一条命令拉起「Streamlit 应用 + MySQL 8.4」，并让**账号 / 馆藏 / 借阅数据、向量库、日志**全部持久化。
> 编写日期：2026-09-19

---

## 一、交付物清单

本次新增 5 个文件，修改 1 个文件：

| 文件 | 类型 | 作用 |
|---|---|---|
| [Dockerfile](Dockerfile) | 新增 | 应用镜像：Python 3.13-slim + 依赖 + 预置知识库 |
| [.dockerignore](.dockerignore) | 新增 | 排除 `.venv`（680MB）与 `.env`（密钥），避免构建上下文膨胀与密钥进镜像 |
| [docker-compose.yml](docker-compose.yml) | 新增 | 编排 app + mysql 两个服务、三个数据卷、健康检查与依赖顺序 |
| [docker/entrypoint.sh](docker/entrypoint.sh) | 新增 | 启动引导：等 MySQL → 幂等建库 → 启动 Streamlit |
| [DOCKER_DEPLOY.md](DOCKER_DEPLOY.md) | 新增 | 本文档 |
| [utils/config_handler.py](utils/config_handler.py) | **修改** | 增加环境变量覆盖机制（见第四节），**默认行为完全不变** |

---

## 二、容器拓扑

```
┌─────────────────────────── Docker 网络（compose 默认 bridge）───────────────────────────┐
│                                                                                        │
│  ┌─────────────────────────────┐          ┌──────────────────────────────────────────┐  │
│  │  library-agent              │          │  library-mysql                           │  │
│  │  (Dockerfile 构建)          │          │  (mysql:8.4 官方镜像)                    │  │
│  │                             │          │                                          │  │
│  │  Streamlit :8501  ←──────┐  │  TCP     │  :3306                                   │  │
│  │  ├ agent/   ReAct 编排    │  │ ───────► │  ├ library_db.users          11 行       │  │
│  │  ├ rag/     Chroma 检索   │  │  3306    │  ├ library_db.books          65 行       │  │
│  │  ├ db/      PyMySQL       │  │          │  ├ library_db.borrow_records 240 行      │  │
│  │  └ model/   DashScope ────┼──┼──────┐   │  └ library_db.borrow_monthly 120 行      │  │
│  │                           │  │      │   │                                          │  │
│  │  卷 state_data → /app/state│  │      │   │  卷 mysql_data → /var/lib/mysql           │  │
│  │     ├ chroma/  向量库      │  │      │   └──────────────────────────────────────────┘  │
│  │     └ md5.txt  去重台账    │  │      │                                                 │
│  │  卷 log_data   → /app/logs │  │      │  宿主机 :8501 ──► 浏览器                       │
│  └───────────────────────────┘  │      │                                                 │
└──────────────────────────────────┼──────┼─────────────────────────────────────────────────┘
                                   │      │
                                   ▼      ▼
                    DashScope 百炼（公网）  仅 HTTPS 出网，无入站要求
                    qwen3-max / text-embedding-v4
```

三个要点：

1. **应用无状态**。容器里不存任何业务数据，随时可 `docker compose up -d --force-recreate` 重建。
2. **MySQL 是硬依赖**，不是可选项：登录校验、工具查库、管理后台三条路径都走它。因此 compose 用 `depends_on.condition: service_healthy` 卡住启动顺序。
3. **向量库走数据卷而非每次重建**。重建要调 DashScope 的 embedding 接口，既有额度成本、又依赖公网，所以镜像里预置了仓库中已构建好的 `chroma_db/`。

---

## 三、快速开始

### 3.1 前置条件

- Docker Engine 20.10+ / Docker Desktop（含 Compose v2，`docker compose` 而非 `docker-compose`）
- **至少 4GB 可用内存**（`chromadb` 加载时占用较多；MySQL 8.4 默认 buffer pool 128MB）
- 出网能力：构建时拉 pip 包，运行时调 DashScope API

### 3.2 配置 `.env`

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# Linux / macOS
cp .env.example .env
```

然后**务必修改** `.env` 中的两项：

```ini
DASHSCOPE_API_KEY=sk-xxxxxx      # 阿里云百炼控制台申请
MYSQL_PASSWORD=<换成一个强口令>    # 这个值同时作为 MySQL root 口令
```

> ⚠️ `MYSQL_PASSWORD` 会被 compose 同时用于两处：MySQL 容器的 `MYSQL_ROOT_PASSWORD`，以及应用连库时的口令。
> 改它等于改数据库 root 口令 —— **数据库已初始化后再改，会导致应用连不上**（见 8.4）。

### 3.3 启动

```bash
docker compose up -d --build
```

预期输出：先构建 app 镜像（首次约 3–6 分钟，取决于网络），再拉起 mysql，等其 healthcheck 通过后启动 app。

打开 <http://localhost:8501> 即可。内置账号（来自 `db/init_db.py` 的种子数据）：

| 账号 | 口令 | 角色 |
|---|---|---|
| `admin` | `admin123` | 管理员（可进后台） |
| `reader1001` … `reader1010` | `123456` | 读者 |

> 生产环境请先在管理后台改掉这些口令，或改 `db/init_db.py` 的 `DEFAULT_ADMIN` / `DEFAULT_READER_PASSWORD` 后清库重跑。

### 3.4 验证部署成功

```bash
# ① 两个容器都在运行，且 app 的 healthcheck 为 healthy
docker compose ps

# ② 应用能连上库：应依次输出 11 / 65 / 240 / 120
docker compose exec app python -c "from db.connection import query_one; [print(t, query_one(f'SELECT COUNT(*) c FROM {t}')['c']) for t in ['users','books','borrow_records','borrow_monthly']]"

# ③ 向量库有内容：应输出 29
docker compose exec app python -c "from rag.vector_store import VectorStoreService; print('vectors =', VectorStoreService().vector_store._collection.count())"
```

---

## 四、配置覆盖机制（本次唯一的代码改动）

### 4.1 问题

`config/db.yml` 里 `host: 127.0.0.1` 是写死的。容器内 `127.0.0.1` 指向**容器自己**，连不到 MySQL 容器，必须改成 compose 的服务名 `mysql`。

有三种解法：

| 方案 | 做法 | 评价 |
|---|---|---|
| a. 改 `config/db.yml` 为 `mysql` | 直接改仓库文件 | ❌ 本地开发（非容器）立刻连不上库 |
| b. bind mount 覆盖配置文件 | 挂一份 `db.docker.yml` 到 `/app/config/db.yml` | ⚠️ 可行，但配置分散在两处，容易改错 |
| c. **环境变量覆盖** | 代码读环境变量，设了就用、没设退回 YAML | ✅ 一个镜像跑遍所有环境，符合 12-Factor |

采用 **c**。

### 4.2 实现

`utils/config_handler.py` 新增一个 12 行的辅助函数：

```python
def _apply_env_overrides(config: dict, mapping: dict) -> dict:
    """用环境变量覆盖配置项，供容器化部署使用。

    YAML 描述的是「默认怎么连」，容器里宿主机名、端口、数据目录往往都不同，
    改 YAML 意味着要往镜像里塞一份环境相关的配置，违背「一个镜像跑遍所有环境」。
    因此这里留出环境变量入口：**设了就用环境变量，没设则完全退回 YAML 原值**，
    本地开发行为不受任何影响。

    :param mapping: {环境变量名: (配置键, 类型转换函数)}
    """
    for env_key, (conf_key, caster) in mapping.items():
        raw = os.environ.get(env_key)
        if raw not in (None, ""):
            config[conf_key] = caster(raw)
    return config
```

在两个加载函数里挂上映射：

```python
# load_db_config
return _apply_env_overrides(conf, {
    "MYSQL_HOST":     ("host",     str),
    "MYSQL_PORT":     ("port",     int),
    "MYSQL_USER":     ("user",     str),
    "MYSQL_DATABASE": ("database", str),
})

# load_chroma_config
return _apply_env_overrides(conf, {
    "CHROMA_PERSIST_DIR": ("persist_directory", str),
    "CHROMA_MD5_STORE":   ("md5_hex_store",     str),
    "CHROMA_DATA_PATH":   ("data_path",         str),
})
```

**口令不在此列**：`MYSQL_PASSWORD` 本来就只从 `.env` 读（`db/connection.py` 里 `os.environ.get`），无需改动。
**类型转换是必要的**：`db.yml` 里 `port: 3306` 是整数，环境变量读出来是字符串。虽然 `connection.py` 里写了 `int(db_conf["port"])` 兜底，但保持类型一致更稳妥。

### 4.3 验证

已在本机 Python 3.13.15 上实测两个方向：

```
=== 无环境变量（应保持 YAML 原值）===
db_conf     = {'host': '127.0.0.1', 'port': 3306, 'user': 'root', 'database': 'library_db', 'charset': 'utf8mb4'}
persist_dir = chroma_db
md5_store   = md5.txt

=== 设置环境变量后（应被覆盖）===
db_conf     = {'host': 'mysql', 'port': 3306, 'user': 'root', 'database': 'library_db', 'charset': 'utf8mb4'}
persist_dir = /app/state/chroma
md5_store   = /app/state/md5.txt
```

`agent.react_agent` / `rag.vector_store` / `db.init_db` 导入均正常。

---

## 五、环境变量总表

### 5.1 应用容器

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 无（**必填**） | 百炼 API Key，`dashscope` 包直接读它。一个 Key 可调全部模型 |
| `CHAT_MODEL_NAME` | `qwen3.8-max` | 覆盖 `config/rag.yml` 的默认对话模型 ID（可选清单仍以 YAML 为准，运行期可在侧边栏切换） |
| `MYSQL_PASSWORD` | 无（**必填**） | MySQL 口令，`db/connection.py` 读它，缺失时抛 `RuntimeError` |
| `MYSQL_HOST` | `127.0.0.1` → compose 设 `mysql` | 覆盖 `config/db.yml` |
| `MYSQL_PORT` | `3306` | 同上 |
| `MYSQL_USER` | `root` | 同上 |
| `MYSQL_DATABASE` | `library_db` | 同上 |
| `CHROMA_PERSIST_DIR` | `chroma_db` → 镜像内设 `/app/state/chroma` | 向量库落盘目录 |
| `CHROMA_MD5_STORE` | `md5.txt` → 镜像内设 `/app/state/md5.txt` | 去重台账路径 |
| `CHROMA_DATA_PATH` | `data` | 知识库源文件目录 |
| `INIT_DB_ON_START` | `1` | 启动时是否执行 `python -m db.init_db`。设为 `0` 可跳过 |
| `DB_WAIT_SECONDS` | `90` | entrypoint 等待 MySQL 的最长秒数 |
| `STREAMLIT_SERVER_PORT` | `8501` | 容器内监听端口 |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | 必须为 `0.0.0.0`，否则只监听回环、宿主机访问不到 |

### 5.2 构建参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `PIP_INDEX_URL` | 清华镜像 | 不在国内时用 `--build-arg PIP_INDEX_URL=https://pypi.org/simple` 覆盖 |
| `PIP_TRUSTED_HOST` | `pypi.tuna.tsinghua.edu.cn` | 与上者配套 |

---

## 六、数据持久化

### 6.1 三个卷

| 卷名 | 挂载点 | 内容 | 删掉的后果 |
|---|---|---|---|
| `state_data` | `/app/state` | Chroma 向量库 + MD5 台账 | 知识库需重建（消耗 DashScope 额度） |
| `mysql_data` | `/var/lib/mysql` | 账号 / 馆藏 / 借阅 | 业务数据全丢，可 `init_db` 恢复种子态 |
| `log_data` | `/app/logs` | 按天切分的 DEBUG 日志 | 仅丢日志 |

### 6.2 为什么向量库和 MD5 台账放进同一个卷

这是本部署方案里最容易踩坑的一处，值得单独说明。

`rag/vector_store.py` 的 `load_document()` 是**只增不减的增量加载**：台账（`md5.txt`）记录过 MD5 的文件会被永久跳过。由此产生一个强耦合：

```
台账在、向量库空   →  load_document() 认为「全都入库过了」，一个文件都不处理
                      →  得到「看起来正常但永远是空的」向量库
向量库在、台账丢   →  load_document() 认为「全是新文件」，重复灌入
                      →  同一段内容在检索结果里出现多次，稀释上下文质量
```

`PROJECT_ANALYSIS.md` 第 6.3 节已经把这个坑写成了警告。容器化时如果给两者分别挂卷（或只挂其中一个），就正好会踩进去。

**解法**：镜像内通过 `CHROMA_PERSIST_DIR` / `CHROMA_MD5_STORE` 把两者**收敛到 `/app/state` 这一个目录下**，compose 只挂 `state_data` 一个卷。两者在物理上再也无法分离。

### 6.3 卷的「首次填充」机制

Docker 有个特性：**命名卷首次被挂载到某个路径时，会用镜像中该路径的内容填充它**。

利用这一点，`Dockerfile` 里的三行把仓库中已构建好的知识库搬到 `/app/state`：

```dockerfile
RUN mkdir -p /app/state/chroma /app/logs \
 && if [ -d chroma_db ]; then cp -a chroma_db/. /app/state/chroma/; fi \
 && if [ -f md5.txt ]; then cp -a md5.txt /app/state/md5.txt; fi
```

于是：

```
docker compose up 首次启动
  └─ state_data 卷为空 → Docker 用镜像内容填充
       └─ /app/state/chroma/   ← 29 条向量，直接可用
       └─ /app/state/md5.txt   ← 5 个文件的台账，与向量库严格对应
```

**收益**：开箱即用，不需要在容器里重跑 `load_document()`，也就不消耗 DashScope 的 embedding 额度。
**副作用**：`state_data` 一旦创建，后续**重建镜像不会更新卷内容**（Docker 只在卷为空时填充）。知识库变更后的正确做法见 7.2。

---

## 七、常用运维操作

### 7.1 查看日志

```bash
# 应用控制台输出（INFO 级，含 agent 工具调用轨迹）
docker compose logs -f app

# 容器内 DEBUG 级日志文件（按天切分）
docker compose exec app sh -c 'ls -l /app/logs && tail -n 100 /app/logs/agent_$(date +%Y%m%d).log'
```

若希望直接在宿主机翻日志文件，把 compose 里的 `log_data:/app/logs` 换成 bind mount：

```yaml
volumes:
  - ./logs:/app/logs          # Linux 下需保证宿主目录对 uid 1000 可写
```

### 7.2 更新知识库

**场景 A：只是改了 `data/` 里的文档内容**

```bash
# 1) 重建镜像（把新的 data/ 和新的 chroma_db/ 带进去）
#    注意：本地仓库里的 chroma_db/ 需要先同步重建一次
.venv\Scripts\python.exe -c "from rag.vector_store import VectorStoreService; vs=VectorStoreService(); vs.reset(); vs.load_document()"
docker compose up -d --build
```

但卷 `state_data` 已经有内容，不会被重新填充。所以还需要：

```bash
docker compose down
docker volume rm langchain-react-agent-main_state_data     # 卷名 = <项目目录名>_state_data
docker compose up -d
```

**场景 B：直接在容器内重建（不重建镜像，更快）**

```bash
# 先在宿主机把新文档放进 data/ —— 但 data/ 没被挂载，所以要 docker cp 进去
docker cp .\data\. library-agent:/app/data/

# 容器内执行全量重建（会消耗 DashScope embedding 额度）
docker compose exec app python -c "from rag.vector_store import VectorStoreService; vs=VectorStoreService(); vs.reset(); vs.load_document()"
```

> 想让 `data/` 改动即时可见，可在 compose 的 app 服务加一行只读挂载：
> `- ./data:/app/data:ro`。这样场景 B 就只剩一条 `exec` 命令，不需要 `docker cp`。

### 7.3 重置数据库到种子状态

```bash
docker compose exec app python -c "from db.connection import get_server_connection; c=get_server_connection(); cur=c.cursor(); cur.execute('DROP DATABASE IF EXISTS library_db'); c.close()"
docker compose exec app python -m db.init_db
```

> ⚠️ 这条会**清空整个库**，包括管理员在后台做的所有改动。只在确认要回到初始种子态时执行。
> 更干净的做法是直接删卷：`docker compose down && docker volume rm <项目名>_mysql_data && docker compose up -d`。

### 7.4 改 MySQL 口令（数据库已初始化后）

改 `.env` 里的 `MYSQL_PASSWORD` 只改了**应用侧**的口令，MySQL 容器内的 root 口令仍是他初始化时的那个（`mysql_data` 卷里存着）。两者不一致 → 应用报 `Access denied`。

正确顺序：

```bash
# 1) 先在库里改口令（用旧口令进）
docker compose exec mysql mysql -uroot -p<旧口令> \
  -e "ALTER USER 'root'@'%' IDENTIFIED BY '<新口令>'; FLUSH PRIVILEGES;"

# 2) 再改 .env
# 3) 重启应用容器
docker compose up -d --force-recreate app
```

### 7.5 备份与恢复

```bash
# 备份数据库
docker compose exec mysql mysqldump -uroot -p"$MYSQL_PASSWORD" --databases library_db > backup.sql

# 恢复
Get-Content backup.sql | docker compose exec -T mysql mysql -uroot -p"$MYSQL_PASSWORD"

# 备份向量库（整个 state 卷打包）
docker run --rm -v langchain-react-agent-main_state_data:/data -v ${PWD}:/backup alpine tar czf /backup/state_backup.tar.gz -C /data .
```

### 7.6 进容器排查

```bash
docker compose exec app bash            # 镜像基于 debian slim，有 bash
docker compose exec app python -c "from db.connection import query_one; print(query_one('SELECT 1'))"
docker compose exec app env | grep -E 'MYSQL|CHROMA|DASHSCOPE'
```

### 7.7 完全清理

```bash
docker compose down              # 停容器，保留数据卷
docker compose down -v           # 连数据卷一起删（数据库 + 向量库全部清空）
docker image rm library-agent:latest
```

---

## 八、排错

| 现象 | 原因 | 处理 |
|---|---|---|
| 应用报 `RuntimeError: 未设置 MYSQL_PASSWORD 环境变量` | `.env` 不存在或键名写错 | 检查 `.env` 在**项目根目录**（compose 只从项目目录读 `.env`） |
| 应用报 `Access denied for user 'root'@'172.x.x.x'` | 口令与 MySQL 容器初始化时不一致 | 见 7.4 |
| `Can't connect to MySQL server on 'mysql'` | MySQL 未就绪，或 `MYSQL_HOST` 被覆盖成了 `127.0.0.1` | `docker compose ps` 看 mysql 是否 healthy；检查 app 的 `environment` 段 |
| 页面能开但登录报错、或 RAG 返回「未检索到」 | 数据库为空 或 向量库为空 | 跑 3.4 的两条验证命令 |
| 中文变问号 `???` | MySQL 字符集不是 utf8mb4 | 确认 compose 里 `--character-set-server=utf8mb4` 生效；已建库需按 7.3 重建 |
| 回答「白屏一段时间后整段出现」 | 流式失效 | 这是 `model/factory.py` 的 `StreamingChatTongyi` 负责的，非容器问题；确认镜像用的是仓库最新代码 |
| 构建卡在 `pip install` | 网络 / 镜像源 | 换源：`--build-arg PIP_INDEX_URL=https://pypi.org/simple --build-arg PIP_TRUSTED_HOST=pypi.org` |
| 构建报 `no such file or directory: docker/entrypoint.sh` | 脚本是 CRLF 换行 | Dockerfile 已用 `sed -i 's/\r$//'` 处理；若仍报错，检查 `.dockerignore` 是否误排除了 `docker/` |
| 宿主机 8501 端口被占 | 本地已有服务 | 改 `ports: "8502:8501"` |
| `docker compose up` 报 `no configuration file provided` | 用了旧版 `docker-compose` v1 | 用 `docker compose`（v2，空格） |

---

## 九、生产化建议

本方案面向**单机演示 / 内部试用**。要上生产，以下五项建议按优先级处理：

1. **轮换 DashScope API Key（最高优先级）**
   `PROJECT_ANALYSIS.md` 第 7.1 节记录的泄露问题**只处理了一半**：
   - ✅ `.env.example` 曾与 `.env` 放着**同一把真 Key**，而它是会被提交进版本控制的模板文件——已换成占位符（2026-09-19）；
   - ⚠️ `.claude/settings.local.json:10` 里那把 Key **仍未清理**，该文件也未被 gitignore；
   - ⚠️ Key 本身**尚未轮换** —— 它已在本地明文落盘，应视为已泄露，须到百炼控制台换一把。

   容器化后 Key 只经 `env_file` 注入、不进镜像层，但**源头那两处仍要处理**。
   另外生产环境建议改用 Docker secret 或外部密钥管理，而非 `env_file`（`docker inspect` 能看到环境变量）。

2. **数据库连接池**
   `db/connection.py` 目前是「每次操作现开现关」（该文件注释已说明演示场景的选择）。容器化后网络往返变成跨容器，开销比本地回环更高。并发上来后建议换 `DBUtils.PooledDB` 或 SQLAlchemy 连接池。

3. **加反向代理与 HTTPS**
   在 app 前面放 Nginx / Caddy 终止 TLS。Streamlit 需要额外配置 WebSocket 转发：
   ```nginx
   location / {
       proxy_pass http://app:8501;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_read_timeout 3600s;      # 长连接，否则对话中途会被切断
   }
   ```

4. **限制端口暴露面**
   `8501:8501` 默认绑定所有网卡。单机场景建议改成 `127.0.0.1:8501:8501`，只允许本机/代理访问。
   同理，MySQL 的 `ports` 在 compose 里已默认注释掉，不要为了图方便放开到 `0.0.0.0`。

5. **给数据卷做定时备份**
   目前 `mysql_data` 和 `state_data` 都只在本地。至少加一条定时任务把 `mysqldump` 输出和 `state_data` 打包推送到异地。

---

## 十、验证状态说明

- ✅ **已验证**：`utils/config_handler.py` 的环境变量覆盖逻辑，两个方向（无变量保持 YAML 原值 / 有变量正确覆盖）均在本机 Python 3.13.15 实测通过；`agent.react_agent`、`rag.vector_store`、`db.init_db` 导入正常。
- ⚠️ **未验证**：`Dockerfile` / `docker-compose.yml` / `entrypoint.sh` **尚未实际构建运行过** —— 编写本文档的机器上未安装 Docker，无法执行 `docker build`。
  因此以下是基于代码与镜像约定的推断，首次部署时请重点核对：
  - `pip install` 在 Python 3.13 下能否全部命中预编译 wheel（`chromadb` 间接依赖 `onnxruntime`，若解析到无 3.13 wheel 的版本会需要编译工具链。届时在 Dockerfile 的 `pip install` 前加 `apt-get install -y build-essential` 即可）；
  - `state_data` 卷的首次填充是否如预期生效（验证方法见 3.4 第 ③ 条，应为 29）。
