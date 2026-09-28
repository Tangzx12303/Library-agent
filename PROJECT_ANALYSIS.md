# 智阅通 · 图书馆智能客服 —— 项目分析文档

> 一个基于 **LangChain 1.x + LangGraph** 的 ReAct Agent 应用：以图书馆业务为场景，集成 RAG 检索增强、MySQL 业务数据源、真实账号体系、运行时 System Prompt 热切换与多模型热切换。
>
> 分析日期：2026-09-25 ｜ 代码版本：本地 `LangChain-ReAct-Agent-main`

---

## 目录

- [一、项目定位与规模](#一项目定位与规模)
- [二、总体架构](#二总体架构)
- [三、四个核心设计决策](#三四个核心设计决策)
- [四、技术栈与选型理由](#四技术栈与选型理由)
- [五、关键实现细节](#五关键实现细节)
- [六、MySQL 搭建与数据层设计](#六mysql-搭建与数据层设计)
- [七、容器化部署](#七容器化部署)
- [八、工程问题记录](#八工程问题记录)
- [九、速查与验证](#九速查与验证)
- [十、对话质检体系与自动化测试](#十对话质检体系与自动化测试)

---

## 一、项目定位与规模

### 1.1 它解决什么问题

读者用自然语言提问，Agent 自主推理并决定"该不该调工具、调哪个工具、下一步做什么"。四类典型请求：

| 请求类型 | 处理方式 |
|---|---|
| **知识问答** | 调 `rag_summarize` 从向量库检索馆内规章资料后总结作答 |
| **馆藏 / 座位 / 时间查询** | 调对应工具，返回结构化信息 |
| **个性化推荐报告** | 走固定工具链（取读者ID → 取月份 → 触发上下文注入 → 取借阅数据 → 查在架可借书目），中途**自动把 System Prompt 换成"报告写手"人格**，输出 Markdown 报告 |
| **多轮追问** | 携带最近若干轮问答，理解「它」「那本书」「再推荐几本」这类指代与省略 |

系统外覆一层**账号体系与管理员后台**：读者注册/登录后以真实身份提问；管理员管理账号、查询任意读者借阅历史、维护馆藏与可借库存。所有业务数据（账号、馆藏、借阅、跨会话记忆）统一落在 **MySQL**。

### 1.2 规模（实测值）

| 指标 | 数值 |
|---|---|
| Python 源文件 | **52** 个（业务 35 + 测试 17） |
| 业务代码 | 总计 **4849** 行 ／ 非空 **3925** 行 ／ 非空非注释 **3548** 行 |
| 测试代码 | 总计 **3089** 行 ／ 非空 **2124** 行 ／ 非空非注释 **1906** 行 |
| 自动化测试 | **286** 个用例，全量运行约 2.5 秒 |
| 工具数量 | 9 个（5 个真实数据 / 3 个 Mock / 1 个信号工具） |
| 检查项 | 11 项（每轮按场景判定其中 4–9 项） |
| 数据库表 | 6 张（`users` / `books` / `borrow_records` / `borrow_monthly` / `conversation_memory` / `eval_turn_records`） |
| 种子数据 | 11 个账号、65 本图书（19 个类别，其中 5 本已借空）、240 条借阅明细、120 行月度汇总 |
| 向量库 | 29 条向量（来自 5 篇知识文档） |
| 配置文件 | 5 个 YAML ｜ 提示词模板 4 个 TXT ｜ 知识库文档 5 个 TXT |
| 可选对话模型 | 8 个（3 个通义 + 5 个第三方，均可热切换） |
| 容器化文件 | 4 个（Dockerfile / docker-compose.yml / .dockerignore / docker/entrypoint.sh） |

代码集中在 4 个文件：`app.py`、`db/init_db.py`、`db/repository.py`、`model/factory.py`，其余均为 250 行以内的小模块。测试与业务代码的比例约 **0.54 : 1**（纯逻辑层基本全覆盖，UI 胶水层不测——理由见第十章）。

---

## 二、总体架构

### 2.1 分层结构

```
┌───────────────────────────────────────────────────────────────┐
│ 表现层   app.py (Streamlit)                                    │
│  · 三视图分发：未登录 → 登录/注册 ｜ reader → 客服 ｜ admin → 后台 │
│  · 侧边栏模型下拉框 → model_id                                  │
│  · st.cache_resource 按 model_id 缓存 Agent                     │
│  · st.session_state 持有消息历史与登录身份                       │
│  · st.write_stream 消费生成器做 token 级渲染                     │
└─────────┬─────────────────────────────────┬───────────────────┘
          │ execute_stream(messages)        │ 管理员后台直连
          │   -> Generator[str]             │ （不经 Agent）
┌─────────▼─────────────────────────────────┼───────────────────┐
│ 编排层   agent/react_agent.py :: ReactAgent(model)             │
│          · create_agent(model, system_prompt, tools, middleware)│
│          · prepare() 组装输入 ／ execute_stream() 流式产出        │
│          agent/memory.py :: build_messages（裁剪 + 压缩）        │
└─────┬──────────────────┬──────────────────┼───────────────────┘
      │                  │                  │
┌─────▼────────┐  ┌──────▼───────────┐  ┌──▼────────────────┐
│ 中间件层      │  │ 工具层            │  │ 数据访问层         │
│ middleware.py│  │ agent_tools.py    │  │ db/repository.py  │
│ · monitor_   │  │ 9 个 @tool        │  │ 15 个 DAO 函数     │
│   tool       │  │ （rag / 馆藏 /    │  │ · 用户管理         │
│ · log_before_│  │   座位 / 时间 /   │  │ · 借阅查询         │
│   model      │  │   身份 / 借阅 /    │  │ · 图书库存         │
│ · report_    │  │   信号工具）       │  │ · 跨会话记忆       │
│   prompt_    │  └────┬──────────┬───┘  └──┬────────────────┘
│   switch     │       │          │         │
└──────────────┘  ┌────▼─────┐ ┌──▼──────┐ ┌▼─────────────────┐
                  │ RAG 层    │ │ 知识库   │ │ MySQL 8.4        │
                  │ vector_   │ │ data/    │ │ （唯一业务数据源） │
                  │ store /   │ │ *.txt    │ │                  │
                  │ rag_service│ └─────────┘ └──────────────────┘
                  └────┬─────┘
┌──────────────────────▼────────────────────────────────────────┐
│ 模型层   model/factory.py                                      │
│  · build_chat_model(model_id) ★ 可热切换                        │
│  · embed_model            固定不变（改了必须重建向量库）          │
└──────────────────────┬────────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────────┐
│ 外部服务  阿里云百炼 DashScope                                    │
│  · 对话：OpenAI 兼容端点（通义 + 第三方统一入口）                 │
│  · 向量化：dashscope SDK 直调（无兼容端点，且不随模型切换）        │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 目录职责

| 目录 | 职责 | 关键约束 |
|---|---|---|
| `agent/` | Agent 的**装配**与**行为扩展** | 不直接写 SQL，只调用 `db.repository` |
| `db/` | 业务数据唯一出口 | 上层不出现裸 SQL；口令只从环境变量取 |
| `rag/` | 向量化与检索问答 | 唯一与 Chroma 交互的地方 |
| `model/` | 模型实例的唯一来源 | 对话模型按 `model_id` 构造；embedding 为模块级单例 |
| `quality/` | **对话质检**：采集、评测、落库、看板聚合与图表 | **分层硬约束**：纯逻辑层不碰 DB，也不 `import streamlit`，因此可被 pytest 直接测 |
| `tests/` | pytest 套件 | 不导入 `app.py`；用 `sys.modules` 桩隔离导入期副作用 |
| `config/` | 参数外置 | 改参数不动代码；**不放口令** |
| `prompts/` | 提示词外置 | 可独立迭代、可版本化对比 |
| `utils/` | 无业务语义的通用能力 | 被所有上层依赖，自身不反向依赖 |
| `data/` | 静态领域知识 + 种子数据 | 运行期只读知识库；CSV 仅作建库种子 |
| `docker/` + 根目录三个 Docker 文件 | 容器化部署 | 不改业务代码；配置差异一律走环境变量 |

`quality/` 内部的依赖方向是单向的，这是它能被测的前提：

```
展示层  render / charts / aggregate   （纯函数，不 import streamlit）
                                            ↑
门面层  store                          （唯一写库处，且永不抛异常）
                                            ↑
逻辑层  evaluator → checks / scenario / recommendation / metrics / trace
                                            ↑
数据    db.repository
```

### 2.3 启动时序与"副作用推迟"

这是本项目一项容易被忽略但价值很高的工程约束——**导入期不做任何重型 IO**：

```
① import app
   ├─ utils.config_handler：load_dotenv() + 加载 5 个 YAML 为模块级单例
   ├─ model.factory：校验 DASHSCOPE_API_KEY 非占位值，并实例化 embed_model
   │                 （唯一在导入期构造的重型对象）
   ├─ agent.tools.agent_tools：仅注册 9 个 @tool 函数  ← 不连 Chroma / MySQL
   └─ db.repository：仅导入函数                       ← 不建立连接
② 渲染登录门（读 st.session_state["user"]）
③ 登录成功：校验口令与账号状态 → 写 session_state
            读者则 start_conversation(reader_id)，从 MySQL 载回跨会话记忆
④ 侧边栏渲染模型下拉框（读 YAML 清单）
⑤ get_agent(model_id) 首次调用：构造对话模型 + 编译 LangGraph 图
⑥ 用户提问：prepare() → execute_stream() → 逐 token 渲染
            └─ 首次调用 rag_summarize 时才构造 RagSummarizeService → 连 Chroma
```

三类重型资源的构造时机被刻意错开：

| 资源 | 构造时机 | 设计意图 |
|---|---|---|
| `embed_model` | 导入期 | 全局唯一、无状态，早构造无副作用 |
| 对话模型 + Agent | 用户选定模型后 | 支持多模型热切换，且按 `model_id` 缓存 |
| Chroma 连接 | 首次真正检索时 | 让 `import agent_tools` 保持零副作用、可独立测试 |
| MySQL 连接 | 每次查询现开现关 | 规避 Streamlit 重跑 + 线程切换带来的连接失效 |

### 2.4 一次「推荐报告」请求的全链路

这是最能体现项目设计的路径。读者 `reader1005`（艺术读者）提问「根据我的借阅数据推荐几本可以马上借到的书」：

| 阶段 | 发生的事 |
|---|---|
| **上下文注入** | `apply_session_context()` 把本段对话固定的 `user_id=1005` 与 `month` 写入 `contextvars` |
| **输入组装** | `prepare()` 清洗历史 → 按 2000 token 预算从新到旧裁剪 → 尾部追加本轮提问 |
| **模型调用 1** | System Prompt = `main_prompt.txt`，其中写有强约束：报告场景必须依次调用 `get_user_id → get_current_month → fill_context_for_report → fetch_external_data → search_books_by_category` |
| **工具链** | 依次执行上述 5 个工具；`fetch_external_data` 查 `borrow_monthly` 表返回偏好类别与在借图书 |
| **状态翻转** | `fill_context_for_report` 被调用这一**事件**被 `monitor_tool` 中间件捕获，置 `context["report"] = True` |
| **约束生效** | `search_books_by_category` 的 SQL 带 `AND available_stock > 0`，读者正在借阅的《艺术的故事》《梵高手稿》（可借=0）**根本不会离开数据库**；返回 4 本可借书 |
| **提示词切换** | 下一次模型调用前，`dynamic_prompt` 中间件读到 `report=True`，System Prompt 换成 `report_prompt.txt` |
| **模型调用 2** | LLM 以「报告写手」人格生成 Markdown 报告，推荐条目只能取自那 4 本 |
| **流式输出** | 仅 `model` 节点产出的 `AIMessageChunk` 被 yield，其余节点与工具返回值被过滤 |

---

## 三、四个核心设计决策

> 这一节是全文重点。项目的技术含量不在"用了 LangChain"，而在这四处**取舍**。

### 3.1 通过工具调用副作用切换 System Prompt

**问题**：同一个 Agent 要服务两种人格——日常客服（简洁问答）与报告写手（长文 Markdown）。把两套规则塞进同一份提示词，会互相干扰，且无法独立迭代。

**方案**：拆成 `main_prompt.txt` 与 `report_prompt.txt` 两份，用 **"信号工具 + 中间件"** 完成运行时切换：

```
main_prompt.txt 写入强约束：报告场景必须调用 fill_context_for_report
        ↓
LLM 遵守约束，调用 fill_context_for_report（函数体是空实现）
        ↓
monitor_tool 中间件（@wrap_tool_call）环绕执行，捕获该工具名
        → request.runtime.context["report"] = True
        ↓
下一次模型调用前，report_prompt_switch（@dynamic_prompt）读到 True
        → 返回 report_prompt.txt 而非 main_prompt.txt
        ↓
LLM 以「报告写手」身份输出
```

**为什么这样设计**：

| 决策 | 理由 |
|---|---|
| 工具只发信号，不写业务 | `fill_context_for_report` 本体什么都不做，**它的价值在于"被调用"这个事件**。给 LLM 一个"意图声明"的出口，业务逻辑交给中间件接管 |
| 切换由数据驱动，而非硬编码 `if` | 是否进入报告模式由 LLM 自己判断并调用工具决定，代码里没有"如果用户说了 XX 关键词就切换"这种脆弱逻辑 |
| 相比"一份提示词塞两个场景" | 两份提示词可独立迭代、独立评估；上下文注入点收敛在中间件，工具函数保持纯净 |
| 相比"在 app 层判断后重建 Agent" | 不重建图、不重连向量库，**零成本**切换；而且切换时机由 Agent 的推理过程决定，前端无从预知 |

`context["report"]` 的生命周期是**单次 `stream()` 调用**——`app.py` 每次提问都新建 `context={"report": False}`，因此不存在跨轮次污染。这是有意为之：仅当将来要支持"记住用户上次要报告"这类跨轮次需求时，才需把状态提升到 `session_state`。

### 3.2 把业务约束编码进工具契约与 SQL，而不只写进提示词

**需求**：推荐的图书必须真实在馆且有足够复本可借。

**难点**：如果只把这句话写进提示词，模型仍可能凭常识编造书目——**提示词是软约束，模型可以违反**。

**方案**：三道防线叠加，强度递增：

| 防线 | 位置 | 做法 | 强度 |
|---|---|---|---|
| ① **数据侧** | `db/repository.py` | `search_books_by_category(only_available=True)` 在 SQL 里加 `AND available_stock > 0`，**可借为 0 的书根本不会离开数据库** | **硬约束** |
| ② 工具契约 | `agent_tools.py` 的 description | 明确"只返回可借图书""生成推荐前必须调用本工具，不得凭常识推荐"，引导模型正确调用 | 软约束 |
| ③ 提示词 | `main_prompt.txt` / `report_prompt.txt` | 规定推荐流程必须包含 `search_books_by_category`，且推荐条目只能取自其返回结果 | 软约束 |

关键在于**防线①**：它不依赖模型是否听话。模型能推荐的书目集合，在数据层面就已被限制为"真实存在且在架可借"；提示词又要求"只能从返回结果中挑选"，两者互相加固。

**类别参数的容错**同样值得说明：读者的"偏好类别"形如 `艺术读者 | 绘画/音乐`，若整串拿去 `LIKE` 则一个类别都匹配不上。因此 DAO 层先按 `[/|、,，\s]` 拆词，再用 OR 连接多个 `category LIKE %s`。这样模型传 `科幻`、`科幻/历史`，还是原样传 `文学爱好者 | 科幻/历史`，都能得到正确结果。

> **可迁移的经验**：凡是"模型可能违反"的业务规则，都应尽量下沉到数据层或接口层，让违反在物理上不可能，而不是在提示词里反复叮嘱。

### 3.3 会话身份固定：`contextvars` + 无状态 Agent

**问题**：`get_user_id` / `get_current_month` 原本每次调用都重新随机取值，导致三类故障：

1. **数据自相矛盾**——LLM 在一次 ReAct 循环里调用两次 `get_user_id`，可能拿到两个不同 ID；
2. **多轮不连贯**——读者追问"再推荐几本"时，"我的借阅数据"换成了另一个人；
3. **不可复现**——同一问题重复提问结果完全不同，无法排查。

**方案**：把这两个值提升为**"一次对话固定一份"的会话上下文**，接入账号体系后 `user_id` 就是**登录读者的真实身份**。

**为什么用 `contextvars` 而不是模块级全局变量？**

`@st.cache_resource` 让**所有 Streamlit 会话共享同一个 Agent 实例**。若用模块级全局变量，A 用户设成 `1003`、B 用户同时在另一浏览器提问，就会读到 A 的值。`contextvars` 的作用域绑定到当前执行上下文，天然隔离并发会话，且是标准库方案。

**为什么每轮都重新注入，而不是"设一次就存住"？**

不依赖 Streamlit 是否复用执行线程——即便每轮换了线程，值也不会丢。持久化交给 `st.session_state`（Streamlit 唯一可靠的会话存储）。

**为什么工具层不 `import streamlit`？**

值通过参数传入 + `contextvars` 中转，工具函数保持与前端框架解耦，可脱离 Streamlit 独立测试。另留有兜底：若脱离 Agent 直接调用工具（脚本调试），`contextvars` 为空时会自动初始化一份随机值，不抛异常。

**同时，Agent 本身保持无状态**。对话历史由调用方（`app.py`）持有并每轮显式传入：

| 对比 | 由 Agent 内部持久化（LangGraph checkpointer + `thread_id`） | 由调用方持有（本项目） |
|---|---|---|
| 会话隔离 | 需为每个 Streamlit 会话分配并维护 `thread_id` | 天然隔离（`session_state` 本就按会话分） |
| 内存增长 | 随会话数**无上限增长** | 随会话结束释放 |
| 可测试性 | 需构造 checkpointer 与 thread | `ReactAgent` 无状态，可独立单测 |
| **多模型切换** | **切换模型 = 换一个 Agent = 上下文丢失** | 新 Agent 拿起同一份消息列表即可无缝续聊 |

最后一行是决定性的：多模型切换让"历史存在 Agent 之外"从一个偏好变成了**必要条件**。

**一个已知的行为层约束（非权限校验）**：`get_user_id` 只能返回**当前登录者**的 ID，模型无法通过工具参数指定去查别人的借阅数据——因为 `fetch_external_data` 的 `user_id` 入参只会从 `get_user_id` 拿到。但要明确：这是**行为层面**的约束，不是安全边界。若要严格防御，应在工具内校验 `user_id == _session_user_id.get()`。当前实现未做此校验，因为管理员后台走的是 `db.repository` 直连路径，不经过 Agent。

### 3.4 用统一端点把"换模型"降维成"换一个字符串"

**问题**：百炼上架的模型分两类，调用协议不同——通义系列可用 DashScope 原生协议，第三方模型（DeepSeek / GLM / Kimi / MiniMax）**只能走 OpenAI 兼容端点**。若沿用 `ChatTongyi`，就得维护两条调用路径。

**方案**：统一收敛到百炼的 OpenAI 兼容端点，对话模型一律用 `langchain-openai` 的 `ChatOpenAI` 构造。

| 收益 | 说明 |
|---|---|
| **切换退化为换字符串** | 通义与第三方走同一条代码路径，`build_chat_model(model_id)` 换个 ID 即可 |
| **丢掉框架适配层** | `ChatOpenAI` 原生支持"流式 + 工具调用"的组合 |
| **清单可外置** | 模型清单写进 `config/rag.yml`，前端据此渲染下拉框，增删模型不动代码 |

> **被删掉的那个补丁值得记一笔**：`ChatTongyi` 在**存在 tools 时**不会开启 `incremental_output`，而是靠"本次全文 − 上次全文"做字符串相减来算增量。但工具调用的 `function.arguments` 是一段 JSON 片段，**对字符串做减法会把它切坏**，导致工具参数解析失败。当时的解法是子类覆盖 `_invocation_params` 强制打开 `incremental_output=True`。改用兼容端点后问题从根上消失——这是一个典型的**换掉底层协议比修补协议更省事**的案例。

**一个模型，三条链路**——这是最容易做错的地方。本项目有**三处**调用 LLM，选中模型必须**同时**作用于这三处，否则会出现"读者看到的是 DeepSeek 写的回答，但检索总结是通义做的"这种不可见的错配：

| 链路 | 注入点 | 传入方式 |
|---|---|---|
| ① 主回答 | `create_agent(model=model, ...)` | `ReactAgent.__init__` 的构造参数 |
| ② RAG 检索总结 | `make_rag_summarize_tool(model)` | **闭包捕获**，透传给 `RagSummarizeService` |
| ③ 历史压缩摘要 | `build_messages(..., model=model)` | `ReactAgent.prepare` 显式传入 `self.model` |

链路②的改造最值得说明。原先 `rag_summarize` 是模块级 `@tool`，`RagSummarizeService` 用模块级变量惰性缓存——意味着"全局只有一个 RAG 服务"，它绑定的是**第一个**用它的 Agent 的模型，多模型下必然错配。改为**闭包工厂**：服务的惰性状态收在闭包里，与工具实例同生命周期，每个模型各一份。

用闭包而非"给工具加一个 `model` 参数"，是因为**模型实例不该让 LLM 看见**——它对读者是透明的实现细节，暴露成入参只会诱导模型去填它。闭包同时保留了原有的**惰性**：导入 `agent_tools` 仍不连接 Chroma。

> 这是本项目里**唯一一处为了多模型而改动的工具代码**。其余 8 个工具与模型无关，一行未动。

**Embedding 刻意不随对话模型切换**（`rag.yml` 里对话模型是"一份清单"，embedding 只有一个名字，这个不对称是有意的）：

| 理由 | 说明 |
|---|---|
| 没有兼容端点 | 百炼的 embedding 只有通义 `text-embedding-*` 系列，未提供 OpenAI 兼容接口，只能由 `dashscope` SDK 直调 |
| 换了会破坏向量空间 | 已有 `chroma_db/` 是用 `text-embedding-v4` 建的。中途换模型，**查询向量与库中向量就不在同一空间**，余弦距离失去意义，检索退化成噪声。真要换必须用 `reset()` + `load_document()` 全量重建 |

---

## 四、技术栈与选型理由

### 4.1 技术栈总表

| 层级 | 技术 | 版本 | 在本项目中的角色 |
|---|---|---|---|
| 语言 | Python | ≥ 3.10（实测 3.13） | 用到海象运算符等特性 |
| Agent 框架 | `langchain` `create_agent` | 1.4.0 | **1.x 新范式**，非 0.3 时代的 `AgentExecutor` |
| 图编排 | `langgraph` | 1.2.11 | `create_agent` 的底层运行时 |
| 中间件 | `langchain.agents.middleware` | 1.4.0 | 工具监控 / 动态提示词 |
| LLM 接入 | `langchain-openai` `ChatOpenAI` | 1.6.2 | 统一经百炼兼容端点接入全部对话模型 |
| 底层 SDK | `openai` | 3.16.2 | `langchain-openai` 的依赖，同时定义鉴权异常类型 |
| LLM | 通义 / DeepSeek / GLM / Kimi / MiniMax | 8 个可选 | 侧边栏可热切换 |
| Embedding | `text-embedding-v4` | `dashscope` 1.27.4 | 文档与查询向量化，**不随对话模型切换** |
| 向量库 | `chromadb` + `langchain-chroma` | 1.5.9 / 1.1.0 | 本地持久化向量存储（HNSW 索引） |
| 文本切分 | `langchain-text-splitters` | 1.1.2 | 中文友好的递归切分 |
| 文档解析 | `pypdf` | 6.18.0 | PDF 文本提取 |
| **关系数据库** | **MySQL** | **8.4** | 唯一业务数据源：账号 / 馆藏（含库存）/ 借阅 / 跨会话记忆 |
| **数据库驱动** | **PyMySQL** | **1.2.0** | 纯 Python 驱动，逐条 SQL，**无 ORM** |
| **认证依赖** | **cryptography** | **50.0.1** | MySQL 8 默认 `caching_sha2_password` 认证所必需（不被代码直接 import） |
| **口令哈希** | `hashlib.pbkdf2_hmac` | 标准库 | PBKDF2-HMAC-SHA256，20 万轮迭代 + 随机盐 |
| 前端 | `streamlit` | 1.63.0 | Web UI 与流式渲染 |
| 配置 | `pyyaml` / `python-dotenv` | 6.0.3 / 1.2.3 | YAML 解析、`.env` 加载 |

### 4.2 关键选型对比

**① 为什么是 PyMySQL 而不是 SQLAlchemy？**

本项目 SQL 总量不大（约 20 条），且都需精确控制——尤其是"可借数量 > 0"这条约束**必须出现在 `WHERE` 子句里**，不能藏在 ORM 的 Python 层过滤中。PyMySQL 只做"连 + 执行 + 取回"三件事，没有会话管理、身份映射、懒加载等概念，出错堆栈也短得多。对这个规模的项目，直接写 SQL 的可读性反而更高。

**② 为什么是 LangChain 1.x 的 `create_agent` 而不是 0.3 的 `AgentExecutor`？**

| 维度 | 旧范式 (0.3) | 新范式 (1.x，本项目) |
|---|---|---|
| Agent 构造 | `AgentExecutor.from_agent_and_tools(...)` | `create_agent(model, tools, ...)` |
| 底层 | 自定义循环 | **LangGraph 状态图** |
| 提示词 | `prompt` 参数 | `system_prompt` 参数 |
| 扩展方式 | 回调 | **Middleware 中间件** |
| 执行 | `.run()` / `.stream()` | 图接口 `.invoke()` / `.stream()` |

**`@dynamic_prompt` 是 1.x 才有的能力，它是本项目"动态提示词切换"得以实现的技术基础**——0.3 时代要达成同样效果，得改写 Agent 内部逻辑。

**③ 为什么用 Streamlit？**

这是个人项目级的快速交付选择：`st.write_stream` 天然支持生成器流式渲染，`st.cache_resource` 解决重型对象缓存，`st.session_state` 解决跨 rerun 状态。代价是**执行模型特殊**——脚本每次交互都"顶到底重新执行"，不是事件回调式。这既是它简洁的原因，也是必须用 `cache_resource` + `session_state` 的原因，下文 5.7 会展开。

**④ 为什么"惰性单例"改成"闭包工厂"？**

见 3.4 链路②。核心结论：**多实例化的需求出现时，模块级单例就是错的**，而闭包是 Python 里承载"每个实例一份惰性状态"最轻的手段。

---

## 五、关键实现细节

### 5.1 Agent 构建与流式输出的节点过滤

`ReactAgent` 对外暴露**两个**方法，这个拆分是刻意的：

| 方法 | 职责 | 是否流式 |
|---|---|---|
| `prepare(query, history, summary)` | 裁剪历史 → 溢出部分压缩 → 返回 `{messages, summary, changed}` | 否（但可能调用模型） |
| `execute_stream(messages, session_context, memory_summary)` | 把组装好的消息送进图，**逐 token 产出正文** | 是 |

拆分原因：**"组装输入"可能触发一次额外的、同步非流式的模型调用**（历史溢出时的压缩摘要）。若与流式产出混在一起，摘要那次调用会先于正文产出，把首字延迟进一步推高。调用方先 `prepare` 拿到结果、把更新后的摘要落库，再把 `messages` 交给 `execute_stream`——这样还有个好处：**摘要的持久化发生在流式渲染之前**，即便读者中途关掉页面，已压缩的记忆也不会丢。

**渲染前的双重过滤**（一个已修复的 Bug，值得单独说明）：

`stream_mode="messages"` 会把**图内所有 LLM 输出**都抛出来，混有两类不该给读者看的噪音：

| 噪音 | 来源 | 后果 |
|---|---|---|
| `ToolMessage` | tools 节点——工具返回值本身 | 读者ID、借阅记录 JSON、RAG 检索原文被拼进回答 |
| `AIMessageChunk` | **工具内部嵌套调用的 LLM**——如 `rag_summarize` 内部的总结链 | RAG 的内部总结被当作回答正文输出 |

第二类尤其隐蔽：它**同样是 `AIMessageChunk`**，只靠类型判断过滤不掉。实测一次报告生成中，两类噪音合计 10 个 chunk 被误输出。

修复方式是**按节点名过滤 + 按类型过滤**，两层缺一不可：

- `metadata["langgraph_node"] != "model"` → 挡掉工具内部的嵌套 LLM 调用；
- `isinstance(message_chunk, AIMessageChunk)` 为假 → 挡掉 `ToolMessage`。

> **修复后仍会看到的内容**：模型在中间轮次输出的思考文本（如"我需要先获取您的读者ID…"）来自 `model` 节点，属于 **Agent 的推理过程**，按项目设计是**有意展示**的（对应 README 的"Agent 推理过程可见"）。

### 5.2 多轮记忆：token 预算裁剪 + 滚动摘要压缩

`agent/memory.py` 解决两件事。

**其一：按 token 预算裁剪，而不是按条数。**

条数不等于长度——一份 Markdown 推荐报告上千 token，一句"谢谢"十几个 token，两者都算 1 条。按条数截断时预算完全不可控。因此：

| 机制 | 参数 | 作用 |
|---|---|---|
| **主阈值** | `max_history_tokens: 2000` | 从新到旧累加，装不下为止 |
| **安全网** | `max_history_messages: 50` | 防止大量极短消息把条数撑到几百条 |
| **最新一条** | — | **无条件保留**：它最相关，被压进摘要得不偿失 |

token 数由**字符启发式**估算（CJK ≈ 1 字符/token，其余 ≈ 4 字符/token），不引入 tokenizer 依赖。它只需要是一个**稳定的预算刻度**，不承担计费职责，因此偏保守即可。

**其二：溢出的历史压缩而不是丢弃。**

被预算挤出去的旧消息交给模型合并进一份**滚动摘要**，随 System Prompt 注入，并按读者 ID 存进 `conversation_memory` 表 —— 因此是**跨会话、跨设备**的：读者换台电脑重新登录也能续上。

摘要的注入点在 `dynamic_prompt` 中间件而非消息列表，原因是**整个会话只该有一条 system 消息**——塞进消息列表会与其 `role` 语义冲突，也会破坏 user/assistant 的交替结构。

| 设计点 | 说明 |
|---|---|
| **清洗** | 逐条校验 `role` 与 `content` 类型，过滤异常结构，避免脏数据污染上下文 |
| **本轮提问最后追加** | 调用方传入的 history **不含**当前提问，避免同一问题被送进模型两次 |
| **失败退回** | 压缩失败只记 warning 并退回上一版摘要——压缩是优化手段，不该让本轮问答中断 |
| **仅在真正变化时落库** | 返回 `changed` 标志，避免每轮都写一次数据库（项目是逐次建连、无连接池） |

`app.py` 侧的数据流是"**先取历史、再追加本轮**"，顺序不能颠倒：

```
st.session_state["messages"]（对话唯一真相来源，同时用于渲染）
    │  ① 先取快照 history = list(...)   ← 此时不含本轮提问
    │  ② 再把本轮提问 append 进去
    ▼
prepare(prompt, history, memory_summary)
    → 清洗 → 按 token 预算裁剪 → 溢出部分压缩成摘要
    │
    └─ changed 为真时 save_memory_summary(reader_id, ...)
```

### 5.3 中间件三件套

位于 `agent/tools/middleware.py`，对应 LangGraph 图的不同节点：

| 装饰器 | 触发时机 | 可做什么 | 项目中的用途 |
|---|---|---|---|
| `@wrap_tool_call` | 每次工具调用前后 | 环绕执行，可改结果 | 日志 + 捕获信号工具以翻转 `context["report"]` |
| `@before_model` | 每次调用 LLM 前 | 观察/改写 `AgentState` | 打日志（`state["messages"]` 条数可用来观察 ReAct 循环推进到第几轮） |
| `@dynamic_prompt` | 每次组装提示词前 | **返回本次使用的 System Prompt** | 场景切换 + 注入长期记忆摘要 |

`monitor_tool` 是**环绕式**的（before/after 都在一处），既能记录入参，也能读取结果，异常时记 error 并重新抛出。`log_before_model` 返回 `None`，表示**只观察、不改状态**。

### 5.4 工具集与 Mock 边界

9 个工具全部用 `@tool(description=...)` 装饰。**description 就是 LLM 的"工具说明书"**，其措辞直接决定 Agent 选工具的准确率，因此写得相当详细（与 `main_prompt.txt` 的工具章节对应）。

| # | 工具 | 入参 | 实现方式 | 性质 |
|---|---|---|---|---|
| 1 | `rag_summarize` | `query` | 工厂函数生成，委托 `RagSummarizeService`（惰性构造，RAG 服务实例收在闭包里） | **真实 RAG** |
| 2 | `search_book` | `book_name` | MySQL `books` 表 LIKE 检索，返回含可借/总复本数 | **真实数据** |
| 3 | `search_books_by_category` | `category` | SQL 层过滤 `available_stock > 0` | **真实数据（约束工具）** |
| 4 | `check_seat` | `area` | 随机数 | 随机 Mock |
| 5 | `get_opening_hours` | 无 | 固定字符串 | 硬编码 Mock |
| 6 | `get_user_id` | 无 | `contextvars` 会话上下文，登录时固定为真实读者 ID | **真实身份** |
| 7 | `get_current_month` | 无 | `contextvars`，建会话时随机取一次 | 随机 Mock（对话内固定） |
| 8 | `fetch_external_data` | `user_id, month` | 查 `borrow_monthly` 表后 `json.dumps` 序列化 | **真实数据** |
| 9 | `fill_context_for_report` | 无 | 空实现 | **信号工具** |

**真实实现 5 个，Mock 3 个，信号工具 1 个**。README 中已逐工具标注，并说明"接入生产环境时替换这些 Mock 即可，Agent 编排逻辑无需改动"。

两个出参规范化细节：

- `fetch_external_data` 的 `json.dumps` 必须带 `ensure_ascii=False`，否则中文被转义成 `\uXXXX`，LLM 读到的借阅数据是一串转义码，严重影响报告质量；
- 数据库中 `borrow_count` / `reading_hours` 是**两个独立的整数列**（便于统计与检索），而提示词与报告模板习惯的是一句话 `月借4册 | 阅读时长21h`，因此在工具层拼回原格式——**出参结构与迁移前的 CSV 版完全一致**，`report_prompt.txt` 无需任何改动。

### 5.5 RAG 管线

**向量库**：`Chroma(collection_name="agent", embedding_function=embed_model, persist_directory="chroma_db")`。传 `persist_directory` 即开启本地持久化，**无需手动调用 `.persist()`**。落盘内容为 `chroma.sqlite3`（元数据 + 文档）与 `{uuid}/data_level0.bin`（HNSW 向量索引）。写入与查询**共用同一个** `DashScopeEmbeddings` 实例，保证向量空间一致。

**分块策略**：`chunk_size=200` / `chunk_overlap=20`，分隔符顺序为 `["\n\n", "。", ".", "?", "？", "!", " ", ""]`。

| 设计点 | 说明 |
|---|---|
| 200 字符偏小（通常 500–1000） | 但中文单字信息密度高于英文单词，配合中文标点分隔符，200 字符约等于一个完整段落，是合理的 |
| 分隔符顺序 | 体现**从粗到细的降级**：先按段落切，切不动再按句号，最后才按空格/字符硬切 |
| 检索 | `as_retriever(search_kwargs={"k": 3})`，取 Top-3 |

**MD5 去重**是这一层最实用的设计。`add_documents` 本身**不去重**——同样的内容重复入库会产生重复向量，检索时 Top-3 可能全是同一段话，稀释上下文质量。项目用"文件级 MD5 台账"做闸门：

| 环节 | 实现要点 |
|---|---|
| 计算 MD5 | 4KB 分片流式读取（海象运算符），避免大文件爆内存 |
| 查重 | 台账**一次性读入 `set`**，查重由 O(n) 线性扫描降为 O(1)（原实现每处理一个文件就重扫整个台账，整体 O(n²)） |
| 主循环 | 单文件失败 `continue`，不中断批量加载；每步空值都记 warning 后跳过 |
| 幂等性 | 已实测：连续两次 `load_document()` 后向量数稳定为 29 |

**局限与逃生舱**：台账是**只增不减**的——它记录"处理过哪些文件"，因此源文件被**删除或改名**后无法自动感知，陈旧向量会一直留在库里。为此提供显式重建入口 `reset()`：清空向量库（`reset_collection()`）与台账后重新加载。

> ⚠️ **向量库与 MD5 台账必须同进同退**：只删向量库而留台账 → 所有文件被认为"已入库" → 得到一个"看起来正常但永远是空的"向量库；反之则重复灌入。这个约束在容器化时是最大的坑（见 7.3）。

**检索问答链**用 LCEL 表达：`PromptTemplate | ChatModel | StrOutputParser`。`|` 运算符把 `Runnable` 串联成管道，因此**整条链仍是 `Runnable`**，天然支持 `.invoke()` / `.stream()` / `.batch()` / `.ainvoke()` 而无需改代码。

上下文拼装时给每段资料编号 `【参考资料N】` 并附带 `metadata`（来源文件、chunk 位置），便于 LLM 溯源。`rag_summarize.txt` 中有一句约束尤其关键——**"纯文本字符串输出，不封装 JSON"**：因为该工具的输出会作为 `ToolMessage` 回灌给 Agent，结构化输出会干扰 ReAct 解析。

### 5.6 配置层与基础设施

**5 个 YAML 在模块导入时一次性加载为全局单例**，所有路径参数均为**相对项目根目录**的相对路径，由 `path_tool` 转绝对路径。`path_tool` 基于 `__file__` 向上推导两级得到项目根，**不依赖 `os.getcwd()`**，因此从任何工作目录启动都能正确定位资源——这是整个项目路径解析的地基。

**`.env` 的加载点只有一处**：`utils/config_handler.py`。因为该模块被所有模块导入，是唯一能保证任何入口（`app.py` / `python -m db.init_db` / 单文件脚本）都拿到环境变量的地方。这里用 `load_dotenv(override=True)` 强制以项目根目录的 `.env` 为准，避免 shell / IDE 里残留的旧值造成 401。

> 曾经 `model/factory.py` 里也有一份重复的 `load_dotenv()`，已删除。当时的隐患是：`python -m db.init_db` 并不导入模型工厂，会拿不到 `MYSQL_PASSWORD`。

**环境变量覆盖机制**（容器化所需，约 12 行）：

YAML 描述的是"默认怎么连"，容器里宿主机名、端口、数据目录都不同。改 YAML 意味着要往镜像里塞一份环境相关的配置，违背"一个镜像跑遍所有环境"。因此 `load_db_config()` / `load_chroma_config()` 在返回前过一道 `_apply_env_overrides()`：**设了就用环境变量，没设则完全退回 YAML 原值**（本地开发行为不受任何影响，两个方向均已实测）。

| 环境变量 | 覆盖的配置键 | 容器内的值 |
|---|---|---|
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_DATABASE` | `db.yml` 对应项 | host 由 `127.0.0.1` → `mysql`（compose 服务名） |
| `CHROMA_PERSIST_DIR` / `CHROMA_MD5_STORE` / `CHROMA_DATA_PATH` | `chroma.yml` 对应项 | 统一收敛到 `/app/state` 下 |
| `CHAT_MODEL_NAME` | `rag.yml` 的 `chat_model_name` | 镜像内**未预设**，按需挂上即可让同一镜像跑不同模型 |

**口令不在此列**：`MYSQL_PASSWORD` 只从 `.env` 读。**列表型配置不做环境变量覆盖**（`chat_model_presets` 只能改 YAML）——列表用环境变量表达易错，而运行期侧边栏切换已足够覆盖容器场景。

> **为什么口令单独放 `.env`？** `config/*.yml` 会被提交进版本控制（它只描述"连到哪"），而口令属于机密（描述"凭什么连"）。两者生命周期不同：换口令不该产生一次代码提交。

**提示词加载（`prompt_loader`）** 三个加载函数统一收敛到内部 `_load_prompt`，做"查配置 → 查缓存 → 读文件 → 双重异常包装"。

为什么需要缓存，以及为什么用 `mtime` 而不是 `lru_cache`：

`dynamic_prompt` 中间件**每次调用模型前都会读一次提示词**，这是高频路径。但直接上 `functools.lru_cache` 又会牺牲"改提示词即时生效"的能力——必须重启应用才能看到改动，调试提示词时非常痛苦。`mtime` 缓存同时拿到两个好处：

| 场景 | 行为 |
|---|---|
| 重复读取同一文件 | 命中缓存，零磁盘 I/O |
| 编辑 `prompts/*.txt` 后再次提问 | `mtime` 变化 → 缓存自动失效 → 重新读取，**无需重启** |

代价是每次读取多一次 `os.path.getmtime()` 系统调用，相比读整个文件可忽略不计。异常处理上区分 `KeyError`（配置项缺失）与 `OSError`（文件读不到），日志能直接指向故障原因。

**日志**采用双 Handler 分级：控制台只出 `INFO`（避免 debug 刷屏），文件记到 `DEBUG`（保留完整链路）——这解释了为什么 `log_before_model` 的详细消息内容用 `logger.debug`。另有**幂等守卫**（`if logger.handlers: return logger`），模块被多次导入时不重复添加 Handler，避免同一条日志打印 N 次。文件名带 `%Y%m%d` 但**没有 RotationHandler**，因此是"每天一个新文件"而非"单文件滚动"。

### 5.7 表现层

`app.py` 按登录状态与角色分发到三个视图：

| 视图 | 内容 |
|---|---|
| 登录/注册 | 两个 Tab；登录校验见 6.9 |
| 读者客服 | 聊天界面 + 侧边栏（模型下拉框 / 快捷提问 / 清空对话 / 长期记忆查看 / 关于） |
| 管理员后台 | 4 个 Tab：用户管理 / 借阅查询 / 图书管理 / 记忆与摘要 |

**Streamlit 的执行模型是理解 `app.py` 的钥匙**：脚本是"**顶到底重新执行**"的，不是事件回调式。所以那些看起来像顶层语句的代码（渲染历史消息的 `for` 循环）每次交互都会重跑——这既是它简洁的原因，也是必须用 `cache_resource` + `session_state` 的原因。

| 机制 | 实现 | 目的 |
|---|---|---|
| **按模型缓存** | `@st.cache_resource` 修饰 `get_agent(model_id)` | 避免每次 rerun 重建 Agent（很贵：要连 Chroma、编译图）；按 `model_id` 分桶，切回用过的模型是秒切 |
| **模型切换** | 侧栏 `selectbox` 读写 `session_state["selected_model_id"]` | 对后续提问立即生效，已渲染的历史不受影响 |
| 会话状态 | `session_state["messages"]` 列表 | 消息历史跨 rerun 保留 |
| **会话身份** | `start_conversation(user["reader_id"])` | 整段对话固定为**登录读者的真实读者 ID** |
| 流式渲染 | `st.write_stream(...)` | 打字机效果 + **返回完整文本**用于落库 |
| 快捷提问 | 侧栏按钮写 `session_state["_pending_prompt"]` | 绕过 `chat_input` 无法程序化赋值的问题 |

**下拉框"值是 ID、展示才是 label"**：`format_func` 让下拉框显示中文名，但内部值与 `session_state` 里存的始终是模型 ID。若把 label 当作值，两个模型 label 重名就会选错，而且 label 一改，存在 `session_state` 里的旧值立刻失效。

`list_chat_model_presets()` / `default_chat_model_id()` 中有两处兜底，都是**为了让下拉框永远可用**而非"配置错了就崩"：

- **清单为空** → 退回"仅含默认模型"的单元素列表。若真返回空列表，`st.selectbox` 收到空 `options` 会直接报错，读者连登录页都进不去；
- **默认值不在清单内** → 退回清单第一项。Streamlit 的 `selectbox` 有个硬性约束：**`key` 对应的值必须出现在 `options` 中**，否则抛异常。配置里改了 `chat_model_name` 却忘了同步 `chat_model_presets` 是很容易犯的错，这里把它降级成"静默退回第一项"而不是整页崩溃。

**快捷提问的实现技巧**：Streamlit 的 `st.chat_input` 不能通过代码设值，因此用一个**中转状态位**——按钮写入 `_pending_prompt`，主流程用 **`.pop()`**（而非 `.get()`）弹出并消费，保证一次性触发，否则下一次 rerun 会重复问同一个问题。

**原文中文化分两层**：`.streamlit/config.toml` 负责**行为**（`toolbarMode = "minimal"` 隐藏 Deploy 按钮、`showSidebarNavigation = false` 关闭多页导航、`gatherUsageStats = false` 关闭匿名统计）；`assets/style.css` 负责**视觉**（隐藏顶栏与三点菜单）。

---

## 六、MySQL 搭建与数据层设计

> 这是项目从"演示 Demo"迈向"可用系统"的关键一层，也是简历上最容易讲清楚的部分。

### 6.1 环境准备与搭建流程

**本地搭建步骤**：

| 步骤 | 操作 |
|---|---|
| 1. 装 MySQL | 8.4 版本（compose 用的 `mysql:8.4` 镜像与之对齐） |
| 2. 配置连接信息 | `config/db.yml`：host / port / user / database / charset（**不含口令**） |
| 3. 配置口令 | `.env` 里设 `MYSQL_PASSWORD`（与 `DASHSCOPE_API_KEY` 同一约定） |
| 4. 建库建表灌种子 | `python -m db.init_db`（幂等，可反复执行） |
| 5. 校验 | 依次输出 `11 / 65 / 240 / 120` 即为就绪 |

**建库脚本 `db/init_db.py` 的两点设计**：

其一，**建库与连接分离**。目标库可能还不存在，所以先要一个"不选库"的连接（`get_server_connection()`，内部 `database=""`）执行 `CREATE DATABASE IF NOT EXISTS`，之后才用常规连接建表。

```sql
CREATE DATABASE IF NOT EXISTS `library_db`
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
```

其二，**幂等策略分级**——这是脚本可以"每次容器启动都跑一遍"的前提：

| 对象 | 幂等策略 | 为什么不用"覆盖" |
|---|---|---|
| 库 / 表 | `CREATE ... IF NOT EXISTS` | 天然幂等 |
| 账号 / 图书 / 月度汇总 | `INSERT IGNORE`（依赖唯一键） | **保留管理员在后台做的修改** |
| 借阅明细 | **表为空时才灌入**（该表无自然唯一键） | 无唯一键可依赖，只能用空表判断 |

### 6.2 五张表设计

| 表 | 主键 | 唯一键 | 用途 |
|---|---|---|---|
| `users` | `id` 自增 | `username`、`reader_id` | 账号与角色（`reader` / `admin`）、状态（`active` / `disabled`） |
| `books` | `id` 自增 | `title` | 馆藏：作者、类别、索书号、馆藏位置、**总复本数 / 可借数量**、状态文案 |
| `borrow_records` | `id` 自增 | 无 | 借阅**明细**：书名、月份、借出/应还/归还日期、状态（`borrowed` / `returned` / `overdue`） |
| `borrow_monthly` | `id` 自增 | `(reader_id, month)` | 借阅**月度汇总**：偏好类别、月借册数、阅读时长、在借图书、对比分析 |
| `conversation_memory` | `reader_id` | — | 跨会话记忆摘要 + `updated_at`（`ON UPDATE CURRENT_TIMESTAMP`） |
| `eval_turn_records` | `id` 自增 | — | **每轮对话的质检记录**：场景、通过/判定项数、推荐与工具指标、耗时、检查项明细 JSON |

`eval_turn_records` 有三个列设计上的讲究，都在第十章展开：比率列一律 `DOUBLE DEFAULT NULL`（不是 `DECIMAL`——pymysql 会返回 `Decimal`，进 `st.dataframe` 时 Arrow 不认，项目已经栽过一次 `ArrowTypeError`）；NULL 表示"不适用"而非 0（否则 `AVG()` 会被知识类问答拉低）；`detail_json` 用 `MEDIUMTEXT` 且**只存派生事实**（检查项结果、书名列表），绝不存工具返回的原始文本。

**表关系（无外键，靠应用层维护）**：

```
users.reader_id ──┬──> borrow_monthly.reader_id   （推荐报告读汇总）
                  └──> borrow_records.reader_id   （管理员查明细）

books.title     ────> borrow_records.book_title   （反规范化存书名）
```

三处值得展开的设计取舍：

| 决策 | 理由 |
|---|---|
| **不用外键** | 演示项目追求部署轻量，且借阅历史与账号是弱耦合关系；约束由应用层 DAO 保证 |
| **`borrow_records.book_title` 存书名而非 `book_id`** | 借阅是**历史事实**。即使某本书后来下架删除了，历史记录仍应完整可读。这是**有意为之的反规范化** |
| **月度汇总与明细拆两张表** | 推荐报告只读汇总（一次查询、字段固定）；管理员查明细时才读 `borrow_records`。避免为一个高频的窄查询扫大表 |
| **`books` 冗余 `available_stock` 与 `status` 文案** | 库存是本项目的核心约束字段，冗余存储避免每次 JOIN 借阅表计算；`status` 文案由 DAO 在改库存时同步维护，避免"可借 0 本却显示在架可借" |

**类别设计**：不采用"文学爱好者 | 科幻/历史"这种**读者画像**作为图书类别，而是拆成 19 个**具体题材**（科幻、历史、哲学、计算机、科普、经济、管理、绘画、音乐、绘本、教育、养生、医学、英语、日语、社会学、心理、小说、传记）——一本《三体》只能属于"科幻"，不能同时属于"科幻/历史"。读者画像到题材的映射由 DAO 的拆词逻辑完成（见 3.2）。

**种子数据里刻意留了 5 本 `available_stock = 0` 的书**（如《沙丘》《人类简史》《艺术的故事》《梵高手稿》《代码大全》）：既是真实场景，也用于验证"推荐图书必须有可借库存"这条约束确实生效。

### 6.3 索引与约束设计

| 表 | 索引 | 服务的查询 |
|---|---|---|
| `books` | `idx_category(category)` | `search_books_by_category` 的 `LIKE` 前缀匹配 |
| `books` | `idx_available(available_stock)` | `available_stock > 0` 过滤 |
| `borrow_records` | `idx_reader(reader_id)` | 按读者查明细 |
| `borrow_records` | `idx_reader_month(reader_id, month)` | 复合索引，覆盖"某读者某月"的查询 |
| `borrow_monthly` | `uk_reader_month(reader_id, month)` | 唯一键，兼作 `INSERT IGNORE` 的幂等依据 |

字符集与排序规则：**全部 `utf8mb4` + `utf8mb4_unicode_ci`**，存储引擎 InnoDB。理由很直接——馆藏书名、作者、偏好类别全是中文，`utf8mb4` 是唯一能完整覆盖中文（含生僻字与 emoji）的选择；compose 里也给 MySQL 服务端显式传了 `--character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci`，否则建表时中文列会退回 latin1，**馆藏书名直接变问号**。

### 6.4 连接层策略

`db/connection.py` 的核心决策是：**每次操作新建连接、用完即关，不做连接池**。

```python
pymysql.connect(
    ...,
    cursorclass=DictCursor,   # 返回 dict 而非 tuple，调用方无需解析列序
    autocommit=True,          # 无需显式 commit
)
```

**为什么不用连接池？** Streamlit 每次交互都会重跑整个脚本，且执行线程可能变化。把 pymysql 连接缓存在模块级或 `st.cache_resource` 里，容易踩到"连接已被服务端断开（`wait_timeout`）"或"跨线程使用"的偶发报错。演示场景并发极低，逐次建连的开销可以接受，换来的是**行为确定**。

> **生产化方向**：改用 `DBUtils.PooledDB` 或 SQLAlchemy 连接池，并补上事务边界（当前是 autocommit，多语句的原子性无保证）。

对外只暴露三个函数，**SQL 参数一律走占位符**（`%s`），杜绝拼接注入：

| 函数 | 用途 | 细节 |
|---|---|---|
| `query_all(sql, params)` | 查询，返回 `list[dict]` | — |
| `query_one(sql, params)` | 查询，返回首个 `dict` 或 `None` | — |
| `execute(sql, params)` | 写操作，返回受影响行数 | 失败时记日志（截断 SQL 到 120 字符）并向上抛 |

每个函数都用 `try/finally` 保证连接关闭。

### 6.5 数据访问层（DAO）

`db/repository.py` 把 SQL 收敛在一处，供 **Agent 工具**、**登录页**、**管理后台**三条路径共用，返回值一律是 `dict` / `list[dict]`。共 **15 个函数**：

| 实体 | 函数 |
|---|---|
| 用户（7） | `get_user_by_username` · `create_user` · `list_users` · `list_readers` · `update_user_status` · `reset_password` · `delete_user` |
| 图书（5） | `search_book_by_title` · `search_books_by_category` · `list_books` · `add_book` · `update_book_stock` |
| 借阅（3） | `get_borrow_monthly` · `get_borrow_monthly_all` · `get_borrow_history` |
| 记忆（3） | `get_memory_summary` · `save_memory_summary` · `list_readers_with_memory` |

> 首版分析时另有 `get_user_by_reader_id`、`update_display_name`、`list_categories`、`get_book_by_id` 四个函数，经全项目 grep 确认**无任何调用点**，已作为死代码删除。

三个 SQL 层面的设计细节：

**① `save_memory_summary` 用 `ON DUPLICATE KEY UPDATE`，而不是"先查再写"**——一是省一次往返，二是并发下不会出现两个请求都判定"不存在"而双双 INSERT 撞主键。传入空串即为清除记忆（清空的是内容，不是行）。

**② `list_readers_with_memory` 用 `LEFT JOIN`**——即使某读者从未生成过记忆，也要出现在列表里（`summary` 为 `None`），让管理员能一目了然地看到"谁有记忆、谁还没有"，而非只看到有记忆的那部分人。

**③ 别名在 SQL 里做中文化**——`get_borrow_monthly_all` 直接 `SELECT month AS 月份, preference AS 偏好类别 ...`，前端 `st.dataframe` 拿到的就是中文表头，无需在 Python 层做映射。

### 6.6 字符集与认证插件：两个必踩的坑

| 坑 | 现象 | 解法 |
|---|---|---|
| **字符集** | 建表时中文列退回 latin1，馆藏书名变问号 | 库、表、连接三处统一 `utf8mb4`；compose 给服务端显式传字符集参数；`pymysql.connect(charset="utf8mb4")` |
| **认证插件** | 报 `RuntimeError: 'cryptography' package is required for sha256_password or caching_sha2_password auth methods` | 装 `cryptography` 包。MySQL 8 默认认证插件是 `caching_sha2_password`，非 SSL 连接下需用 RSA 公钥加密口令，而 PyMySQL 的这部分实现依赖它 |

第二项容易被误判——`cryptography` **不被项目代码直接 import**，看起来像多余依赖，但它是 PyMySQL 连 MySQL 8 的硬依赖。所以它在 `requirements.txt` 里是显式列出的。

**为什么选 PyMySQL 而非 `mysqlclient` / `mysql-connector-python`**：PyMySQL 是纯 Python 实现、无 C 扩展，`pip install` 不需要编译器——在 Windows 开发和 slim 容器镜像里这点很关键。

### 6.7 账号与口令安全

口令**不存明文**，用标准库的 PBKDF2-HMAC-SHA256 加盐哈希，存储格式 `salt_hex$hash_hex`：

| 要点 | 说明 |
|---|---|
| **每口令独立随机盐**（16 字节，`secrets.token_bytes`） | 相同口令产生不同哈希，防彩虹表；盐随哈希一起存储 |
| **20 万轮迭代** | 兼顾安全与登录响应速度（约几十毫秒量级），暴力破解成本足够高 |
| **`hmac.compare_digest` 常量时间比较** | 避免通过响应时间差逐字节猜出哈希。若用 `==`，它在首个不同字节处短路返回，攻击者可测量响应时间逐字节推断 |
| **不引入 bcrypt / argon2** | `hashlib.pbkdf2_hmac` 是标准库，少一个需要编译的二进制依赖，部署更简单 |
| **脏数据返回 False 而非抛异常** | 登录流程只关心"能否通过"，不该因一条脏数据导致整个页面崩溃 |

**登录校验的完整链路**（`app.py`）：

| 次序 | 判断 | 失败提示 |
|---|---|---|
| 1 | 按用户名查用户 + 校验口令哈希 | **同一条**"用户名或密码错误" |
| 2 | `status == "active"` | "该账号已被禁用，请联系管理员" |
| 3 | 通过 | 只把 `id / username / role / display_name / reader_id` 写进 `session_state` |

第 1 步的"同一条提示"是刻意的：**不泄露用户名是否存在**，避免账号枚举攻击。第 3 步则保证**口令哈希不出数据库层**。

### 6.8 常用运维 SQL 与命令

```sql
-- 观察库存分布（推荐约束的数据基础）
SELECT category, COUNT(*) AS 品种, SUM(available_stock > 0) AS 可借品种
FROM books GROUP BY category ORDER BY 品种 DESC;

-- 查看某读者的月度汇总
SELECT month, preference, borrow_count, reading_hours, current_books
FROM borrow_monthly WHERE reader_id = '1005' ORDER BY month;

-- 查看跨会话记忆的覆盖情况
SELECT u.reader_id, u.display_name, LENGTH(m.summary) AS 记忆长度, m.updated_at
FROM users u LEFT JOIN conversation_memory m ON m.reader_id = u.reader_id
WHERE u.role = 'reader' ORDER BY m.updated_at DESC;
```

| 操作 | 命令 |
|---|---|
| 建库建表灌种子（幂等） | `python -m db.init_db` |
| 校验各表行数 | `python -c "from db.connection import query_one; [print(t, query_one(f'SELECT COUNT(*) c FROM {t}')['c']) for t in ['users','books','borrow_records','borrow_monthly','conversation_memory','eval_turn_records']]"` → 前四项应输出 `11 / 65 / 240 / 120`；后两项随实际使用增长（`eval_turn_records` 每轮对话 +1） |
| **质检数据的口径自检** | `python -c "from db.connection import query_one; r=query_one('SELECT COUNT(*) t, COUNT(rec_f1) n, AVG(rec_f1) a FROM eval_turn_records'); print(r)"` → `n < t` 说明"不适用"确实落成了 NULL 并被 `AVG` 跳过 |
| **跑自动化测试** | `python -m pytest`（286 个用例，约 2.5 秒） |
| **从零重建数据库** | 先用 `get_server_connection()` 执行 `DROP DATABASE IF EXISTS library_db`，再 `python -m db.init_db` |

> ⚠️ 重建会**清空整个库**，包括管理员在后台做的所有改动。只有确认"要回到初始种子状态"时才执行。

---

## 七、容器化部署

> 完整操作步骤（备份 / 恢复 / 排错 / 生产化建议）见 [DOCKER_DEPLOY.md](DOCKER_DEPLOY.md)，本节只记录**设计决策与它对代码的影响**。

### 7.1 拓扑与启动顺序

```
docker compose up -d --build
   │
   ├─ 构建 library-agent 镜像（Python 3.13-slim + 依赖 + 预置知识库）
   ├─ 拉起 mysql:8.4 ──► healthcheck 通过（能真正跑通 SELECT 1）
   └─ 启动 app ──► entrypoint.sh
                     ├─ 等 MySQL 端口可连（兜底，防止 docker run 单跑时抢跑）
                     ├─ python -m db.init_db   ← 幂等，每次启动都跑是安全的
                     └─ streamlit run app.py   ← 监听 0.0.0.0:8501
```

宿主机只需映射 `8501`；MySQL 的 3306 在 compose 里**默认不对外暴露**（注释掉的 `ports` 段），需要 Navicat 直连时再放开——一旦暴露到 `0.0.0.0`，root 口令就是唯一防线。

**healthcheck 用"能真正跑通一条查询"而非 `mysqladmin ping`**：ping 在权限/认证异常时也可能返回成功。compose 的 `depends_on: condition: service_healthy` 保证应用启动时库已就绪，而 `entrypoint.sh` 里的端口等待是对"单独用 `docker run` 起容器"这一场景的兜底。

### 7.2 为容器化改动的业务代码

只有一处机制，覆盖三组配置项：**环境变量覆盖**（实现见 5.6），约 12 行，**默认行为不变**。

三种可选解法里选了它，而非"改 YAML"或"bind mount 一份 docker 专用配置"，理由是**"一个镜像跑遍所有环境"**：

| 方案 | 问题 |
|---|---|
| 改 `config/db.yml` 为 `host: mysql` | 本地开发就断了，且环境相关信息进了版本库 |
| bind mount 一份 docker 专用 YAML | 多一份需要同步维护的配置副本 |
| **环境变量覆盖**（选用） | 镜像与环境解耦；不设变量时行为与改造前完全一致 |

> 多模型切换这一特性**没有为容器化额外付出成本**：默认模型本就是个配置项，需要时挂上 `CHAT_MODEL_NAME` 即可；而运行期在侧边栏换模型是纯前端行为，与部署方式无关。

### 7.3 三个数据卷与"首次填充"机制

| 卷 | 挂载点 | 内容 |
|---|---|---|
| `mysql_data` | `/var/lib/mysql` | 数据库文件 |
| `state_data` | `/app/state` | **向量库 + MD5 台账**（同卷） |
| `log_data` | `/app/logs` | 按天切分的日志文件 |

**为什么向量库与 MD5 台账必须放进同一个卷？** 这是容器化时最容易踩的坑。5.5 说过 `load_document()` 是只增不减的增量加载，台账与向量库互为前提：

```
台账在、向量库空   →  load_document() 认为「全都入库过了」，一个文件都不处理
                      → 得到「看起来正常但永远是空的」向量库
向量库在、台账丢   →  load_document() 认为「全是新文件」，重复灌入
                      → 同一段内容在检索结果里出现多次，稀释上下文质量
```

容器里若给两者分别挂卷、或只挂其中一个，就正好会踩进去。解法是通过 `CHROMA_PERSIST_DIR` / `CHROMA_MD5_STORE` 把两者**收敛到 `/app/state` 这一个目录**，compose 只挂 `state_data` 一个卷——**两者在物理上再也无法分离**。

**卷的"首次填充"机制省掉了重新向量化**：Docker 的命名卷首次被挂载到某路径时，**会用镜像中该路径的内容填充它**。于是 Dockerfile 在构建时把仓库里已构建好的 `chroma_db/` 与 `md5.txt` 复制到 `/app/state`，首次以空卷启动时自动填进去。

| 收益 | 说明 |
|---|---|
| 开箱即有 29 条向量 | 不需要在容器里重跑 `load_document()` |
| 不消耗 API 额度 | 重新向量化会消耗 DashScope 的 embedding 额度 |
| **副作用** | 卷一旦创建，重建镜像不会更新它。知识库变更后的正确流程是"先本地重建 → 重建镜像 → 删卷"，详见 DOCKER_DEPLOY.md 第 7.2 节 |

### 7.4 构建优化与安全

| 关注点 | 处理 |
|---|---|
| **依赖层缓存** | 先 `COPY requirements.txt` 再 `pip install`，最后才 `COPY . .` —— 改代码不会触发重装依赖 |
| **`.venv/`（680MB）** | `.dockerignore` 排除。不排除的话每次 build 都要打包上传 680MB，且会覆盖镜像里装好的依赖 |
| **密钥** | `.env` 与 `.claude/` 被 `.dockerignore` 排除，不进镜像层；运行时经 compose 的 `env_file` 注入 |
| **`chroma_db/` + `md5.txt`** | **故意不排除**，它们是 7.3 预置机制的数据来源 |
| **国内网络加速** | `PIP_INDEX_URL` 作为 build arg，可覆盖为官方源 |
| **时区** | `tzdata` 让日志时间戳与宿主机一致（slim 镜像默认没有时区库，只设 `TZ` 会退回 UTC） |
| **运行身份** | 非 root（uid 1000）。应用不需要写 `/app` 之外的内容 |
| **探活** | 用 `python -c "urllib.request.urlopen(...)"` 打 `/_stcore/health`，省掉在镜像里装 curl |
| **CRLF** | `sed -i 's/\r$//' docker/entrypoint.sh` —— 在 Windows 上编辑过的脚本可能带 CRLF，会让 `/bin/sh` 报 `no such file or directory` |

### 7.5 常用命令

```bash
# 首次部署（先复制 .env.example 为 .env 并填入两个凭据）
docker compose up -d --build
docker compose ps
docker compose logs -f app

# 容器内执行（迁移库表 / 重建知识库）
docker compose exec app python -m db.init_db
docker compose exec app python -m rag.vector_store

# 停机（保留数据卷）/ 连数据卷一起清掉
docker compose down
docker compose down -v
```

### 7.6 与安全事项的关联

容器化**不解决** 8.2 节所述的 Key 泄露问题——密钥只是从"明文写死在文件里"变成"运行时注入环境变量"，源头那个已泄露的 Key 仍须轮换。且环境变量注入本身也比 Docker secret 弱一档：`docker inspect` 能看到值。

---

## 八、工程问题记录

### 8.1 已修复的代码缺陷

| # | 位置 | 原问题 | 修复方式 |
|---|---|---|---|
| 1 | `utils/file_handler.py` | 路径不存在时返回 `("txt","pdf")` 这个**后缀字符串元组**，与正常路径的**文件路径元组**语义相反。调用方会拿 `"txt"` 去算 MD5——属典型**静默错误**（类型注解捕获不到） | 改为 `return ()` |
| 2 | `fetch_external_data` | 声明 `-> str`，实际返回 `dict`，与工具描述不符 | 显式 `json.dumps(..., ensure_ascii=False)` |
| 3 | `rag/vector_store.py` | `__main__` 块是遗留调试代码，查询词与图书馆知识库无关 | 改为正式 CLI 入口 `python -m rag.vector_store` |
| 4 | `agent_tools.py` | 模块级 `rag = RagSummarizeService()` **导入期即连 Chroma**，任何 `import agent_tools` 都产生副作用 | 改为闭包工厂 `make_rag_summarize_tool(model)` 惰性构造 |
| 5 | `rag/vector_store.py` | MD5 台账每个文件都重扫一遍，整体 O(n²)；文件删除后台账无法清理 | ① 台账一次性读入 `set`，查重降为 O(1)；② 新增 `reset()` 用于全量重建 |
| 6 | `utils/prompt_loader.py` | 每次调用都读磁盘（`dynamic_prompt` 是高频路径），无缓存 | 按 `mtime` 缓存，文件变更自动失效 |
| 7 | `react_agent.execute_stream` | **工具返回值混进读者可见的回答**：`stream_mode="messages"` 抛出图内所有 LLM 输出，含 `ToolMessage` 与工具内部嵌套的 `AIMessageChunk` | 按 `langgraph_node == "model"` + `isinstance(AIMessageChunk)` 双重过滤 |
| 8 | `db/init_db.py` | **借阅数字被拼接**：用 `"".join(ch for ch in text if ch.isdigit())` 抽数字，把 `"月借4册 \| 阅读时长21h"` 抽成 `"421"` 而非 `4` 和 `21`——120 行种子数据全部错误 | 先按 `\|` 切分，再在含目标单位的片段里取数字；并写一次性脚本修复已灌入的 120 行 |
| 9 | `app.py`「清空对话」 | **清空会把读者身份也清掉**：`new_session_context()` 无参时随机挑一位读者，于是每次清空后 Agent 面对的都是另一个人 | `start_conversation(reader_id)` 显式传入登录读者的 ID；随机兜底仅在未传参时生效 |
| 10 | `fetch_external_data` | **越权查询**：`user_id` 入参由 LLM 填写，工具内不校验它是否等于当前登录读者 | 新增 `_authorize_reader()`；**直读 `_session_user_id.get()` 而非 `_session_value`**（后者会随机兜底，会把一道鉴权变成抛硬币） |
| 11 | 同上 | **鉴权过了但查询查询带空格**：模型传 `"  1005  "`，比较前 strip 了，但传给 SQL 的是原值；MySQL 的 `=` 对前导空格敏感，静默查空 | 规范化提前到使用之前，`user_id`/`month` 都 strip 后再用 |
| 12 | `quality/charts.py` | **图表配置被静默丢弃**：`configure(padding=...)` 会**重置整个 config**，挂在链尾把前面设好的网格线、坐标轴配色、字体、图例配置全冲掉，图表退回 Altair 默认外观 | 把它挪到配置链**首位**；用测试钉住 |
| 13 | `quality/recommendation.py` | **表格排版的书单整表漏判**：模型用 Markdown 表格荐书，单元格写「4 本」而非「可借 4 本」，一次完全正确的推荐被判成召回率 0 | 补列表/表格形态识别（详见 10.5） |
| 14 | 同上 | **表格里的编造书目漏检**：上一条的判据被我错误地绑在了"书名在候选集内"上，而编造的书恰恰不在候选集里——最该抓的情形反而漏掉 | 补一条不要求候选集成员资格的规则，判据改用**索书号** |
| 15 | `app.py` 错误提示 | 数据库连不上也会提示「请检查 `DASHSCOPE_API_KEY`」，把排查方向带偏 | `_friendly_error()` 补 pymysql 分支 |

> **第 7 项是新增功能时才暴露出来的老 Bug**：原代码每次只问一个问题、工具调用少，泄漏内容混在长回答里不易察觉；多轮对话下工具调用变多，泄漏才变得显眼。
>
> **第 8、9 项是改造中实测发现的**：第 8 项是比对"数据库读出的值"与"CSV 原值"时才暴露出 `421` 这个不可能的数；第 9 项是把随机身份改成真实登录身份后顺带暴露的——原来随机的时候，"清空后换个人"看起来是特性，改成真实账号后就成了 Bug。

### 8.2 安全问题（**部分处理，Key 轮换待办**）

`DASHSCOPE_API_KEY` 曾在**三个**位置以明文出现：

| 位置 | 是否被 gitignore | 状态 |
|---|---|---|
| `.env` | ✅ 已覆盖 | 保留原值（本地运行必需） |
| `.env.example` | ❌ **未覆盖，会进版本库** | ✅ 已修复：换成占位符（**它恰恰是三者中最危险的**——会被提交） |
| `.claude/settings.local.json` | ❌ 未覆盖 | ✅ 已修复：删除含 Key 的权限条目 |

**仍待处理**：到百炼控制台**轮换该 Key**——它已在本地明文落盘，应视为已泄露。这一步只能由本人完成。

> 好在当前目录**还不是 git 仓库**，尚未提交推送。但一旦 `git init` + `commit`，修复前的两个文件都会进入版本历史，届时清理成本会高得多。

### 8.3 有意保留（未修改）

| 项 | 说明 |
|---|---|
| **提示词与工具描述重复维护** | `main_prompt.txt` 重述了 9 个工具的能力边界，与 `@tool(description=...)` 内容重叠。**这是有意保留的**：System Prompt 中的工具说明权重高于工具自身的 description，当前措辞是针对"提升工具选择准确率"调优过的。用脚本自动生成会改变措辞，可能引入行为回退 |
| **`context` 与 `session_state` 的边界** | `context["report"]` 是**单次 `stream()` 调用的作用域**，不存在跨轮次污染——**这个设计是正确的，无需修改** |
| **测试只覆盖纯逻辑层** | `app.py` 的胶水代码（约几十行）没有测试。导入它会执行 `st.set_page_config` 和整段视图分发，无法在测试里安全 import。这是**有边界、已知的代价**；视图层靠 Streamlit 的 `AppTest` 做过无头渲染验证（不纳入 pytest 套件，因为需要真实数据库） |

> 「缺少自动化测试」原先是这一节里的第一条，现已补齐（286 个用例），详见第十章。

### 8.4 依赖版本说明

`langchain-community` 停留在 **0.4.x**（而非 1.x）是正常的——community 包的版本线与核心包不同步。改造为兼容端点后它只剩两个用途：`DashScopeEmbeddings` 与 `PyPDFLoader` / `TextLoader`（`ChatTongyi` 已不再使用）。

多模型改造带来的依赖变化：

| 包 | 变化 | 原因 |
|---|---|---|
| `langchain-openai` | **新增** | 提供 `ChatOpenAI`，接入百炼兼容端点 |
| `openai` | **新增** | `langchain-openai` 的底层 SDK，同时定义鉴权异常类型 |
| `langchain-community` | 用途收窄 | 原先还用于 `ChatTongyi`，现只剩向量化与文档加载 |
| `dashscope` | 用途收窄 | 原先对话与向量化都走它，现只服务 embedding |

数据库相关新增两个：`PyMySQL`（纯 Python 驱动）与 `cryptography`（MySQL 8 认证硬依赖，见 6.6）。口令哈希**没有新增依赖**。

---

## 九、速查与验证

### 9.1 改动地图

| 想改什么 | 改哪里 |
|---|---|
| **增删可切换的对话模型** | `config/rag.yml` 的 `chat_model_presets`（加一行即可；清单在导入期加载，**需重启**） |
| 改默认选中的对话模型 | `config/rag.yml` 的 `chat_model_name`（须同时在清单内，否则退回第一项） |
| 换 Embedding 模型 | `config/rag.yml` 的 `embedding_model_name`，**改完必须重建向量库** |
| 换百炼地域 / 兼容端点 | `model/factory.py` 的 `DASHSCOPE_COMPATIBLE_BASE_URL`（新加坡地域改 `dashscope-intl`） |
| 调检索 Top-K、分块大小 | `config/chroma.yml` |
| 改 MySQL 连接地址 / 端口 / 库名 | `config/db.yml`（容器内可用环境变量覆盖） |
| 改 MySQL 口令 | `.env` 的 `MYSQL_PASSWORD`（**不要写进 `db.yml`**） |
| 增删馆藏图书 / 改库存 | 管理员后台「图书管理」标签页，或改 `db/init_db.py` 的 `SEED_BOOKS` 后重跑建库脚本 |
| 改初始账号 / 默认口令 | `db/init_db.py`（**改完需清库重跑才生效**，`INSERT IGNORE` 不会覆盖已有账号） |
| 改「可借才推荐」的约束 | `db/repository.py` 的 `only_available` 参数 + `prompts/report_prompt.txt` 的输出规则 4 |
| 加一个可查询的字段 / 表 | `db/init_db.py` 加 DDL，`db/repository.py` 加 DAO 函数 |
| 改 Agent 人格 / 工具说明 | `prompts/main_prompt.txt`（**改完无需重启**，mtime 缓存自动失效） |
| 改推荐报告格式 | `prompts/report_prompt.txt` |
| 改 RAG 总结风格 | `prompts/rag_summarize.txt` |
| 加一个新工具 | `agent_tools.py` 加 `@tool`，再到 `react_agent.py` 的 `tools=[...]` 注册 |
| 加一个新中间件 | `middleware.py` 加装饰器函数，再到 `react_agent.py` 的 `middleware=[...]` 注册 |
| 换知识库文档 | 往 `data/` 放 `.txt`/`.pdf`，重跑 `load_document()` |
| 重建知识库（删除了源文件） | `VectorStoreService().reset()` 后再 `load_document()` |
| 改多轮历史长度 | `config/agent.yml` 的 `max_history_tokens`（主阈值）与 `max_history_messages`（安全网），任一设为 0 = 关闭多轮 |
| 关闭记忆压缩 | `config/agent.yml` 设 `memory_summary_enabled: false` |
| 改记忆压缩的提示词 | `prompts/memory_summary.txt`（含 `{previous_summary}` / `{new_messages}` 占位符） |
| 改模型调用失败时的错误提示 | `model/factory.py` 的 `describe_dashscope_error()` + `app.py` 的 `_friendly_error()` |
| **改质检的检查项** | `quality/checks.py` 的 `ALL_CHECKS`（加一个返回 `CheckResult` 的函数即可） |
| **改场景判定规则 / 必需工具组** | `quality/scenario.py` 的 `SCENARIOS` 与 `_PATTERNS`（改完记得同步 `tests/test_scenario.py`） |
| **改推荐书目的识别规则** | `quality/recommendation.py`（改完先跑 `tests/test_recommendation.py`，那里有真实模型输出做样本） |
| 改看板的图表或聚合口径 | `quality/charts.py` / `quality/aggregate.py`（都不含 Streamlit，可直接单测） |
| 改小样本阈值 / 看板筛选默认值 | `quality/aggregate.py` 的 `MIN_SAMPLES_FOR_RANKING`；筛选默认值在 `app.py` 的 `_tab_eval()` |
| **加/改自动化测试** | `tests/`（新增文件直接放进去即可，`pytest.ini` 已配 `testpaths`） |
| 改 UI 样式 | `assets/style.css` + `.streamlit/config.toml` |
| 改快捷提问示例 | `app.py` 的 `EXAMPLES` 列表 |
| 容器化：改端口 / 卷 / 服务名 | `docker-compose.yml` |
| 容器化：改启动引导 | `docker/entrypoint.sh` + `INIT_DB_ON_START` 环境变量 |

### 9.2 验证清单

**① 验证多轮上下文**

以 `reader1005` / `123456` 登录，连续输入：

| 轮次 | 输入 | 预期表现 |
|---|---|---|
| 1 | 图书馆有哪些功能分区？ | 正常回答 |
| 2 | **那自习区呢？** | 能理解省略的主语，直接介绍自习区，而非反问「您指哪个」 |
| 3 | 根据我的借阅数据生成推荐报告 | 生成报告，数据为 `reader1005`（登录身份） |
| 4 | **再推荐几本科幻类的** | 引用第 3 轮的数据 + 本轮新偏好；两轮引用的读者数据完全一致 |

点「清空对话」后再提问：消息与月份重置，但**读者身份保持不变**。

**② 验证「在馆可借」约束**

| 方法 | 操作 |
|---|---|
| 最快的 | `reader1005` 的历史在借图书《艺术的故事》《梵高手稿》`available_stock = 0`。提问推荐时，这两本**不应出现在推荐书目区**，但允许作为"您正在借阅的书"被说明性提及 |
| 手动改库存 | 管理员把某本书可借数量改成 0 → 读者提问该书类别 → 不再出现；改回 > 0 后又回来 |
| 直接查 SQL | 调 `search_books_by_category('艺术读者 \| 绘画/音乐')`，去掉 `only_available` 再看一次——已借空的书**只在这一次**出现，这正是"约束写在 SQL 的 `WHERE` 里"的直接证据 |

**③ 验证多模型切换**

| 步骤 | 预期 |
|---|---|
| 看侧边栏模型下拉框 | 默认选中 `rag.yml` 的 `chat_model_name`（当前 `qwen3.8-max`） |
| 切到另一模型后**接着追问** | 新回答仍能理解省略的主语（**历史没丢**）；「关于本项目」里的"当前对话模型"已变 |
| 切回用过的模型 | 响应明显更快（缓存命中，未重建 Agent 与 Chroma 连接） |
| 改 `chat_model_presets` 增删一项但不重启 | 下拉框**不会**变化——YAML 在导入期已固化为模块级单例。改清单需重启；而"选哪个模型"是运行期状态，不需要重启 |

**常见误判**：切换后报错**不代表切换机制有问题**，更可能是所选模型本身不可用。界面的错误提示就是为此设计的——按提示区分 `invalid_api_key`（Key 失效）、`AccessDenied`（权限受限）、`ModelNotExist`（模型未开通）、`FreeTierOnly`（免费额度用尽）与"不支持工具调用"五种情况。**排查顺序建议：先换回默认模型确认基线可用**，再定位所选模型的具体问题。

---

## 十、对话质检体系与自动化测试

> 这一章记录的是"怎么知道系统答得好不好"——在 LLM 应用里，这个问题比"怎么让它答"更难。

### 10.1 为什么需要它

在做这套东西之前，改提示词、换模型都只能靠**人工看几条回答**来判断效果。这有两个后果：没有客观依据，而且**无法发现退步**——某次提示词改动让报告场景的召回掉了一半，肉眼看不出来。

目标定得很明确：**让质量可量化、可展示、可回归**。具体拆成三件事——修掉一个越权缺陷、补一套 pytest、建一套在线质检让读者和管理员都能看到质量数据。

### 10.2 指标口径（**最容易被做错的地方**）

#### ① 「工具选择准确率」的直觉定义是错的

最自然的想法是：用关键词规则推导「本轮该调用哪些工具」，再拿实际调用去比，算 precision/recall。**这个定义会系统性地把好回答判成坏的**：

| 读者提问 | 规则期望 | 模型实际做的 | 按直觉定义算出的准确率 |
|---|---|---|---|
| 图书馆有哪些功能分区？ | `{rag_summarize}` | 调了 `rag_summarize` **和** `get_opening_hours`（分区开放时间确实相关） | **50%** |
| 有哪些科幻类的书可以借？ | `{search_books_by_category}` | 先 `search_book("三体")` 核实，再查类别 | **50%** |

两例中模型都做了比期望更好的事，却各被扣一半分。**指标衡量的是"模型是否恰好和关键词表一样懒"，而不是"答得好不好"。**

改为三个各自有可靠 ground truth 的数：

| 指标 | ground truth | 定义 | 为什么不会误伤 |
|---|---|---|---|
| `tool_recall`（必需工具召回率） | **提示词自身的契约** | 每个场景定义若干 **OR 组**，命中组内任一即算满足；得分 = 满足组数 / 总组数 | OR 组让"多调一个同类工具"不被惩罚 |
| `tool_precision`（调用合规率） | **禁用集**，而非期望集 | `|实际 ∩ 允许| / |实际|`，`允许 = 全部工具 − 禁用` | 只惩罚提示词**明令禁止**的调用；绝大多数场景禁用集为空 → 恒为 1.0 |
| `extra_call_rate` | 无 | `|实际 − 必需| / |实际|` | **不参与评分**，只进图表——"该模型话痨"值得看见，但不该算错 |

实质规则只有一条：非报告场景禁用 `fill_context_for_report`（`main_prompt.txt` 明确写了"绝对不调用"）。

#### ② 关键词表必须窄，且误判不能产生扣分

场景判错会**假扣分**：读者问「自习区的开放时间」，若规则判成 `seat`，模型正确地调了 `get_opening_hours`，必需组 `{check_seat}` 没人满足 → 召回率变 0。

对策有三层：

- **关键词只收没有第二种读法的词**，且顺序上做避让（`开放时间` 排在 `自习区` 之前）。刻意**不收**裸的「推荐」二字——「再推荐几本科幻类的」是报告对话的追问，不该被要求重走一遍完整报告流程。
- **匹配不上就落到 `other`，而 `other` 的必需组为空 → 指标是"不适用"**。所以误判**永远不会产生错误扣分**，只会少算覆盖。
- **`scenario_source` 落库**（`trace` / `keyword` / `unknown`），`unknown` 占比在看板上显示——**这是用来审计关键词表本身、而不是信任它的**。

还有一条刻意的选择：**轨迹里出现 `fill_context_for_report` 时不改判场景**。它是模型自己声明的事实，但若关键词判定是别的场景，就该让"未误触发报告流程"这条检查报警——那正是我们要看见的缺陷，不能糊过去。

#### ③ 推荐准确率有个会被提示词直接触发的漏洞

`report_prompt.txt` **明确指示**模型写这样一句话：

> 读者历史借阅过的书若当前已借空（可借数量为 0），只能说明「该书当前已借空」并推荐同类别下可借的替代书目

若直接正则抓 `《》` 里的书名，已借空的《艺术的故事》会被从这句**否定句**里抓出来当成一次推荐，再对着正确排除了它的候选集算成**假阳性**——**模型严格遵守了提示词，却被判失误**。

处理方式：

1. **按子句切分再分类**，不按行。逗号必须切：「《艺术的故事》已借空，建议改借《西方美术史》」若整体处理，会把本该推荐的《西方美术史》一起否掉。
2. **三类归属**：命中否定语气 → `negated`（分子分母都不进）；命中可借宣告/推荐标题/结构化列表 → `recommended`；其余 → `unclassified`（不计入指标，面板上显示"另有 N 处提及未计入"）。
3. **候选集取自轨迹，不重新查库**。它就是工具返回的、被 SQL 的 `available_stock > 0` 过滤过的书目。质检要判断的是"模型有没有用好喂给它的数据"，所以必须用它**实际收到**的那份——否则库存在此期间一变，就是拿一把变化的尺子量一个固定的回答。
4. **"无编造书目"做成二元检查项而非比率**。一本编造的书混在十本好书的推荐里，准确率只掉几个点，信号会被稀释。

**两个由真实对话暴露并修掉的漏判**（详见 10.5）——一个是表格排版的书单整表漏判，一个是编造书目放进表格后逃过检测。

#### ④ 不适用一律存 NULL，不存 0

「图书馆有哪些功能分区？」没有图书推荐，推荐 P/R 就是**不适用**，不是 0。

- 表里所有比率列 `DOUBLE DEFAULT NULL`；
- UI 用 `rate(None) == "—"`，并在面板下方写明「— 表示本轮不适用，不计入统计」；
- **关键在于**：MySQL 的 `AVG()` 会自动跳过 NULL。若落 0，每个知识类问题都会把推荐 F1 的均值往下拽，问答类型分布一变，看板上就出现一次纯属虚构的塌陷。

通过率同理：分母是 `passed / determined`（**判定过的项数**），不是 `passed / 11`。一轮普通问答只判定 4–8 项，拿固定总数当分母会让通过率随场景漂移，横向比较失去意义。

### 10.3 工具调用轨迹怎么采集

复用项目里**已有且已验证**的通道：`monitor_tool` 里 `request.runtime.context["report"] = True`，已经证明「中间件写 `runtime.context` → 下游中间件读得到」。

在 LangGraph 源码里逐层确认了这条通道成立：`create_agent` 的 `context_schema` 默认是 `None`，`_coerce_context` 因此把传入的 dict **原样返回**；`Runtime.override` 用 `dataclasses.replace` 但不覆盖 `context` 字段，所以同一次 `stream()` 内**所有 `wrap_tool_call` 共享同一个 dict 对象**。

两个必须处理的坑：

| 坑 | 后果 | 对策 |
|---|---|---|
| **一次模型回复里的多个工具调用是并发执行的**（`tool_node.py` 用 `ContextThreadPoolExecutor`） | `list.append` 的顺序**不等于**模型的调用顺序 | 每条记录带一个模块级 `itertools.count()` 产生的 `seq`，读取时按 `seq` 排序 |
| **失败的工具调用会丢痕** | "模型调了工具但工具炸了"从质检视野里消失，等价于把失败算成没发生 | 条目在 `handler(request)` **之前**登记，异常路径回填 `error` 后重新抛出 |

`react_agent.py` 的 `context={...}` 字面量上留了注释：这个 dict **有意保持无 schema**——将来若有人给 `create_agent` 传 `context_schema`，`report` 和 `tool_calls` 会一起失效。

> 顺带发现并钉进测试的一条机制：**`StructuredTool.invoke()` 运行在上下文的副本里**，所以 `_session_value` 里的 `var.set()` 不会回流到调用方（直接调用函数则可以）。这不影响生产路径（`execute_stream` 总会先绑定身份，工具对上下文只读），但值得留一条测试，免得后人写出依赖"兜底值会被固化"的代码。

### 10.4 失败隔离：质检是旁路，绝不能影响对话

四层防护，每层都有具体理由：

| 层 | 做法 | 理由 |
|---|---|---|
| 门面 | `quality/store.py` 的每个入口都吞异常 | 落库失败**也要返回报告**——存不存得下和读者看不看得到是两回事 |
| 检查项 | 每项单独 try/except，失败记为"不适用" | 一条正则写错不该让整份报告消失 |
| 渲染 | 面板整块包 try/except | 避免重蹈项目早期 `ArrowTypeError` 的覆辙 |
| 崩溃轮 | 在**已有的 `except` 分支**里补一条记录 | 见下 |

**第四层是最容易被忽略的一层。** 工具抛异常时 LangGraph 会中断整轮执行，`response` 为 `None`，质检那一行根本不会写。结果是**数据库越坏，管理员的看板看起来越好**——失败轮次全部消失了。

所以要显式补一条 `scenario='failed'` 的记录（所有指标列为 NULL），看板才有真实的"异常中断"计数，通过率的分母也才是诚实的。

顺带修了一个既有小问题：数据库错误原本也会显示「请检查 `DASHSCOPE_API_KEY`」（`describe_dashscope_error` 对 pymysql 异常返回 `None`，落到兜底文案）。现在补了 pymysql 分支。

### 10.5 由真实对话暴露的两个漏判

这两条都是**先跑一轮真实对话、再拿真实输出做回归样本**才发现的，值得记下来：

**其一：Markdown 表格排版的书单整表漏判。** 模型把可借书目排成：

```
| 序号 | 书名 | 作者 | 索书号 | 可借数量 |
| 1 | 《流浪地球》 | 刘慈欣 | I247.5/1201 | 4 本 |
```

单元格里写的是「4 本」而不是「可借 4 本」，行内可用性规则没命中，于是**一次完全正确的推荐被判成"什么都没推荐"，召回率记成 0**。补的规则是：书名在候选集内且落在列表项/表格行 → 算推荐。

**其二：编造书目放进表格后逃过检测。** 上面那条规则我最初写成了"书名在候选集内 **且** 是表格行"，于是**编造的书恰恰因为不在候选集里而漏掉了**——最该抓的情形反而抓不到。修法是补一条**不要求候选集成员资格**的规则，判据改用**索书号**（`report_prompt.txt` 要求推荐书目必须标注索书号，因此它是个可靠的"在向读者交付可取书目"信号）。

为什么不用"只要是列表就算"，是因为那会误伤另一类真实回答——「您当前在借：- 《艺术的故事》」这种列表不该被算成推荐。索书号恰好把两者分开了。

四条形态现在都有回归测试钉住（含一条用**真实模型输出**做的样本）：表格推荐 → P/R 均 1.0；说明性提及 → 不适用；否定句 → 只认真正的推荐；表格里的编造书 → 被抓出。

### 10.6 检查项清单

11 项，每项返回三态（**通过 / 未通过 / 不适用**）。三态是必须的：「本轮没有推荐」时的"推荐书目均标注索书号"是**不适用**，算成通过会虚高通过率，算成未通过会冤枉模型。

| 检查项 | 判定 | 不适用条件 |
|---|---|---|
| 回答非空 | 回答有非空白内容 | 从不 |
| 未直出工具原始返回 | 不含 `当前在架可借图书共` / `"偏好类别"` 等特征串 | 从不 |
| 未在预算内空答 | 非（工具调用数 < 5 且回答含「我不知道」） | 从不 |
| 无越权查询 | 所有 `fetch_external_data` 的入参与登录读者一致 | 本轮未查借阅记录 |
| 无编造书目 | 推荐书目 ⊆ 工具返回的在架可借书目 | 无推荐书目 |
| 推荐书目均标注索书号 | 每本推荐书的所在行含索书号 | 无推荐书目 |
| 同书未同时推荐与否定 | 两个集合不相交 | 无推荐书目 |
| 报告流程完整 | 轨迹含 `fill_context_for_report` + `fetch_external_data` + `search_books_by_category` | 非报告场景 |
| 报告含在馆声明 | 含 `report_prompt.txt` 要求的收尾声明 | 非报告场景 |
| 未误触发报告流程 | 轨迹不含 `fill_context_for_report` | 本身就是报告场景 |
| 座位数据来自工具 | 轨迹含 `check_seat` | 非座位场景 |

### 10.7 管理员看板

数据源**只有**读者真实对话的在线质检记录，不是标准化测试集——这句话写在图表旁边，否则模型之间的小样本差异会被当成结论。

顶部一行**全局筛选**（时间范围 / 模型 / 场景），作用于下方所有图表；每张图配一个表格双胞胎。

| # | 形式 | 内容 | 为什么是这张图 |
|---|---|---|---|
| 0 | 指标卡，**不画图** | 质控轮次 / 全项通过率 / 推荐 F1 均值 / 工具召回均值 / 异常中断 / 场景未识别率 | 一个数就是一个数，画成图是浪费。caption 里写明分母 |
| 1 | 横向条形图，单色 | 检查项失败分布（降序，条末直标百分比） | **全页信息量最高的一张**：它直接指出该改提示词的哪一句 |
| 2 | 分组横向条形图，双色 | 按模型的工具召回率 vs 全项通过率 | 指导侧边栏换模型这个核心动作。**n < 5 的模型不参与排名**——那是运气不是质量 |
| 3 | **两张图共享 x 轴、上下堆叠** | 上：通过率折线（末点直标）；下：每日轮次 | 通过率离了样本量读不懂：1 轮全过和 40 轮全过都是 100% |
| 4 | 热力图（单色蓝顺序阶） | 场景 × 检查项 通过率 | **发现规则错误的唯一手段**：某场景整列同一项失败时，先怀疑必需工具组定义错了 |

**明确不做的**：单条"平均分"趋势线（受问答类型分布污染）、小样本模型排名、场景占比饼图、**双 y 轴**、失败率的颜色渐变（单序列就该单色）。

配色上 app 自身的主色 `#3a6ea5` 不动（不重绘应用），图表标记用规范的中性分类色板，按固定槽位取用、不循环：本模块只用得到前两槽（蓝 `#2a78d6` / 橙 `#eb6834`），热力图用单一蓝色顺序阶（`#cde2fb` → `#0d366b`）。

> ⚠️ **色板校验器没能运行**：规范要求用 `scripts/validate_palette.js` 跑一遍对比度与色盲可辨性检查，但本机**没有安装 Node.js**。当前用法是文档里已验证通过的相邻两槽，且每张图都配了表格双胞胎与直接标注；若要在图表里扩展到第三、第四槽，应先补跑校验器。

### 10.8 自动化测试

**286 个用例，全量 2.5 秒**。分层原则是：**纯逻辑层全覆盖，UI 胶水层不测**。

#### 测试隔离：不能靠环境变量

直觉做法是在 `conftest.py` 里 `os.environ.setdefault("DASHSCOPE_API_KEY", ...)`，但它**解决不了问题**：`utils/config_handler.py` 在导入期就调用 `load_dotenv(override=True)`，`.env` 里那份值会**覆盖**掉刚设的值。于是当某台机器的 `.env` 还停留在占位符时，`_require_dashscope_key()` 抛错，**整个测试套件在收集阶段就崩**。

本机 `.env` 存的是真实 Key，所以恰好能跑——**这是最坏的一类 bug：本地看不见，CI 上必炸**。改环境变量治不了，必须往 `sys.modules` 里预注册桩模块（`model.factory`、`rag.rag_service`），在任何业务模块导入之前。

顺带挡掉了整条 `chromadb` 的导入，收集阶段快了好几秒。

**绝不导入 `app.py`**：导入它会执行 `st.set_page_config`、`load_css()` 和整段视图分发。所以可测逻辑全部放在 `quality/` 里（`render` / `charts` / `aggregate` 都不含 Streamlit），`app.py` 只留胶水。

#### mock 要打对目标

`db/repository.py` 是 `from db.connection import query_all, query_one, execute`，所以必须打 **`db.repository.query_all`**。打 `db.connection.query_all` **毫无效果**——测试会在"通过"的同时真的连上数据库。这是本项目最容易踩的 mock 陷阱，已在 `conftest.py` 的 `no_db` 夹具里写明。

#### 覆盖面

| 文件 | 覆盖 |
|---|---|
| `test_security_fetch_external_data.py` | 越权拒绝/放行，且 spy 断言仓库层**从未被调用** |
| `test_security_password.py` | 口令哈希正反例、脏存储串不抛异常 |
| `test_trace.py` | 轨迹记录、**失败路径留痕**、并发下的次序还原、中间件接线 |
| `test_metrics.py` | P/R/F1 的 None 传播与分母为 0 |
| `test_scenario.py` | 每场景关键词、OR 组、**模糊提问落到 other**、重叠避让 |
| `test_recommendation.py` | 否定句不污染、表格/列表识别、编造书目、**真实模型输出样本** |
| `test_checks.py` | 每项检查的三态 + 单项失败不影响整体 |
| `test_evaluator.py` | 端到端 + NULL 落库形态 |
| `test_eval_repository.py` | SQL 参数化、不适用落 NULL、失败隔离 |
| `test_aggregate.py` | 小样本保护、不适用不参与均值、跨轮累加 |
| `test_charts.py` | 配色取用规则、**无双 y 轴**、配置链不被覆盖 |
| `test_memory.py` / `test_session_context.py` / `test_repository.py` / `test_file_handler.py` | 既有模块的纯逻辑与边界 |

#### 测试抓出的三个真实缺陷

写测试的过程中抓到三处**代码**问题（不是我写错测试）：

| 缺陷 | 症状 | 根因 |
|---|---|---|
| 鉴权通过但查询带空格 | 模型传 `"  1005  "`，鉴权 strips 后放行，但**传给 SQL 的是原值**。MySQL 的 `=` 对前导空格敏感，查询静默返回空 → 读者看到"未查到借阅记录" | 规范化只做在比较处，没做在使用处 |
| 图表配置被静默丢弃 | 网格线、坐标轴配色、字体、图例样式**全部失效**，图表退回 Altair 默认外观（粗网格、带边框）。不报错，只是变丑 | `.configure(padding=...)` 会**重置整个 config**，必须放在配置链**首位** |
| 表格里的编造书目漏检 | 见 10.5 其二 | 结构化列表的判据错误地绑在了候选集成员资格上 |

### 10.9 运行方式

```bash
python -m pytest            # 286 个用例，约 2.5 秒
python -m pytest tests -q   # 静默模式
python -m pytest tests/test_recommendation.py -v   # 单个文件
```

---

## 附：与面试问答的对应关系

配套的面试技术问答见 [INTERVIEW_QA.md](INTERVIEW_QA.md)。两份文档的分工是：

| 文档 | 定位 |
|---|---|
| **本文** | 客观描述"项目是什么、怎么做的、为什么这么做" |
| **INTERVIEW_QA.md** | 面向面试场景，把本文的技术点组织成"被问到怎么答、追问怎么接" |
