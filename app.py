"""智阅通 · 图书馆智能客服 —— Streamlit 应用入口。

三种视图由登录身份决定：
  - 未登录      → 登录 / 注册页
  - role=reader → 读者智能客服（ReAct Agent 对话）
  - role=admin  → 管理后台（用户管理 / 借阅查询 / 图书管理）

说明：
- 右上角原生 Deploy / 三个点工具栏已通过 .streamlit/config.toml + assets/style.css 隐藏；
- 中文化的菜单与操作统一收敛到左侧边栏；
- 左侧边栏被「锁定」为常驻面板（initial_sidebar_state="locked"），不可折叠/关闭。
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

import pymysql
import streamlit as st

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import new_session_context
from db import repository as repo
from model.factory import (build_chat_model, default_chat_model_id,
                           describe_dashscope_error, list_chat_model_presets)
from quality import aggregate as quality_aggregate
from quality import charts as quality_charts
from quality import render as quality_render
from quality import scenario as quality_scenario
from quality import store as quality_store
from utils.logger_handler import logger
from utils.security import verify_password

# ---------------------------------------------------------------
# 页面基础配置
# ---------------------------------------------------------------
# initial_sidebar_state="locked"（Streamlit 1.63 原生支持）：
#   桌面端 —— 侧边栏强制展开，折叠按钮不渲染、切换函数为空操作，
#             且前端不再读取 localStorage 里遗留的折叠标记，
#             因此「关掉就再也打不开」的状态不会出现；
#   窄屏端 —— 锁定自动降级为可折叠（否则侧边栏会盖住正文），
#             此时由 style.css 保留的「展开侧边栏」按钮负责收回。
st.set_page_config(
    page_title="智阅通 · 图书馆智能客服",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="locked",
)

# ---------------------------------------------------------------
# 自定义样式
# ---------------------------------------------------------------
CSS_PATH = Path(__file__).parent / "assets" / "style.css"


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css()

# ---------------------------------------------------------------
# 全局缓存：模型 / 工具 / RAG 服务等重型对象只构建一次
# ---------------------------------------------------------------
# 以 model_id 为缓存键：每个模型各缓存一份 Agent（各自持有自己的 RAG 服务与
# 惰性连接）。这样在侧边栏切换模型时，已用过的模型不必重建，切回去是秒切。
@st.cache_resource(show_spinner=False)
def get_agent(model_id: str) -> ReactAgent:
    return ReactAgent(build_chat_model(model_id))


def model_selectbox() -> str:
    """侧边栏的模型下拉框，返回当前选中的模型 ID。

    下拉框的**值**是模型 ID，展示用的是 label（format_func），这样两个模型
    即使 label 重名也不会选错。选中结果落在 session_state["selected_model_id"]，
    跨 rerun 保持；初始值取自配置的默认模型，且保证一定在选项列表内，
    否则 Streamlit 会因「key 的值不在 options 中」直接报错。
    """
    presets = list_chat_model_presets()
    model_ids = [p["id"] for p in presets]
    labels = {p["id"]: p["label"] for p in presets}

    if st.session_state.get("selected_model_id") not in model_ids:
        st.session_state["selected_model_id"] = default_chat_model_id()

    st.selectbox(
        "🧠 对话模型",
        options=model_ids,
        key="selected_model_id",
        format_func=lambda model_id: labels.get(model_id, model_id),
        help="切换后对该对话的后续提问生效；RAG 检索总结与历史压缩会用同一个模型",
    )
    return st.session_state["selected_model_id"]


# ---------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def start_conversation(reader_id: str) -> None:
    """开启一段新对话：清空消息，并为本段对话固定上下文（读者ID + 月份）。

    读者身份取自登录信息，全程不变；月份重新随机，等同于换一个统计口径重新开始。
    跨会话记忆摘要则从数据库载回——消息列表清空了，但「记得」的东西要续上。
    """
    st.session_state["messages"] = []
    st.session_state["session_context"] = new_session_context(reader_id)
    # 跨会话记忆只是锦上添花，加载失败（如 conversation_memory 表尚未建好）
    # 不应阻断读者登录——降级为「无记忆」继续进入会话，而不是让登录页直接崩溃。
    try:
        st.session_state["memory_summary"] = repo.get_memory_summary(reader_id) or ""
    except Exception as e:
        logger.warning(f"[登录]读取跨会话记忆失败，已降级为无记忆：{e}")
        st.session_state["memory_summary"] = ""
    st.session_state.pop("_pending_prompt", None)


def header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">📚 智阅通 · <span class="grad">图书馆智能客服</span></div>
            <div class="app-subtitle">基于 ReAct 推理范式与检索增强生成（RAG），实时回答读者咨询</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===============================================================
# 质检面板（读者侧）
# ===============================================================
def render_quality_panel(compact: dict, *, expanded: bool = False) -> None:
    """把一份精简质检报告渲染成折叠面板。

    :param compact: ``EvaluationReport.to_compact()`` 的产出
    :param expanded: 侧边栏的「最近一轮」默认展开，聊天区里的默认折叠
    """
    if not compact:
        return

    # 整块包一层 except：这里出问题（比如 Arrow 序列化）绝不能影响读者看到回答
    try:
        with st.expander(f"✅ 本轮质检 · {quality_render.headline(compact)}",
                         expanded=expanded):
            if compact.get("error"):
                st.error(f"本轮对话未正常完成：{compact['error']}")

            columns = st.columns(3)
            for column, (label, value, help_text) in zip(
                    columns, quality_render.metric_cards(compact)):
                column.metric(label, value, help=help_text)

            rows = quality_render.checks_rows(compact)
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)

            st.caption(f"工具调用轨迹：{quality_render.tools_text(compact)}")
            st.caption(quality_render.caption(compact))

            with st.popover("📐 指标口径说明"):
                st.markdown(quality_render.scope_note())
                st.caption(
                    f"场景：{compact.get('scenario_label', '—')} ｜ "
                    f"模型：{compact.get('model_id') or '—'} ｜ "
                    f"耗时：{compact.get('latency_ms') or '—'} ms"
                )
    except Exception as e:                                  # noqa: BLE001
        logger.error(f"[质检]结果渲染失败（对话不受影响）：{e}")


def _friendly_error(exc: Exception, model_id: str) -> str:
    """把异常翻译成读者能照着做的提示。

    鉴权类报错交给 ``describe_dashscope_error`` 去区分 401/403。它返回 None 时，
    原先一律落到「请检查 DASHSCOPE_API_KEY」那句兜底文案 —— 于是**数据库连不上
    也会被提示去查 API Key**，把排查方向整个带偏。这里补一个数据库分支。
    """
    described = describe_dashscope_error(exc, model_id)
    if described:
        return described

    if isinstance(exc, pymysql.MySQLError):
        return (
            "数据库连接失败，无法生成回复。请确认 MySQL 已启动、`.env` 中的 "
            "`MYSQL_PASSWORD` 正确，且 `config/db.yml` 的连接信息无误。"
        )

    return (
        "模型调用失败，无法生成回复。请检查 `.env` 中的 `DASHSCOPE_API_KEY` "
        "是否有效（申请地址：https://bailian.console.aliyun.com/ ）。"
    )


def timed_stream(generator, box: dict):
    """包装生成器，记录首字延迟与总耗时。

    首字延迟（TTFB）是读者真正感知到的那个数 —— 对一个流式界面来说，
    它比总耗时更有信息量。
    """
    started = time.perf_counter()
    try:
        for chunk in generator:
            if box.get("ttfb_ms") is None:
                box["ttfb_ms"] = int((time.perf_counter() - started) * 1000)
            yield chunk
    finally:
        box["latency_ms"] = int((time.perf_counter() - started) * 1000)


def logout_button() -> None:
    if st.sidebar.button("🚪 退出登录", width="stretch"):
        st.session_state.pop("user", None)
        st.session_state.pop("session_context", None)
        # 只清内存里的副本；跨会话记忆已落库，下次登录会重新载回
        st.session_state.pop("memory_summary", None)
        st.session_state["messages"] = []
        st.rerun()


# ===============================================================
# 视图一：登录 / 注册
# ===============================================================
def render_login() -> None:
    header()

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        login_tab, register_tab = st.tabs(["🔑 登录", "📝 注册读者账号"])

        # ---------------- 登录 ----------------
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="如 admin 或 reader1001")
                password = st.text_input("密码", type="password")
                submitted = st.form_submit_button("登录", type="primary", width="stretch")

            if submitted:
                user = repo.get_user_by_username(username.strip())
                if user is None or not verify_password(password, user["password_hash"]):
                    st.error("用户名或密码错误")
                elif user["status"] != "active":
                    st.error("该账号已被禁用，请联系管理员")
                else:
                    st.session_state["user"] = {
                        "id": user["id"],
                        "username": user["username"],
                        "role": user["role"],
                        "display_name": user["display_name"],
                        "reader_id": user["reader_id"],
                    }
                    if user["role"] == "reader":
                        start_conversation(user["reader_id"])
                    st.rerun()

        # ---------------- 注册 ----------------
        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("用户名", placeholder="登录用的账号")
                new_display = st.text_input("昵称", placeholder="选填，如「小王」")
                new_password = st.text_input("密码", type="password")
                confirm = st.text_input("确认密码", type="password")
                reg_submitted = st.form_submit_button(
                    "注册", type="primary", width="stretch"
                )

            if reg_submitted:
                new_username = new_username.strip()
                if not new_username or not new_password:
                    st.error("用户名与密码不能为空")
                elif new_password != confirm:
                    st.error("两次输入的密码不一致")
                elif len(new_password) < 6:
                    st.error("密码长度至少 6 位")
                elif repo.get_user_by_username(new_username) is not None:
                    st.error("该用户名已被占用")
                else:
                    try:
                        # 读者ID默认取用户名；新账号暂无借阅历史，
                        # 智能客服会如实告知「未检索到借阅记录」。
                        repo.create_user(
                            username=new_username,
                            password=new_password,
                            role="reader",
                            display_name=new_display.strip() or new_username,
                        )
                        st.success(f"注册成功，请使用 {new_username} 登录")
                    except Exception as e:
                        st.error(f"注册失败：{e}")

        st.caption(
            "演示账号 —— 管理员：admin / admin123；"
            "读者：reader1001 ~ reader1010 / 123456（这些账号带完整借阅数据）"
        )


# ===============================================================
# 视图二：读者智能客服
# ===============================================================
EXAMPLES = [
    "图书馆有哪些功能分区？",
    "帮我查一下《三体》的馆藏情况",
    "根据我的借阅数据进行个性化推荐",
    "有哪些科幻类的书可以借？",
]


def render_chat(user: dict) -> None:
    reader_id = user["reader_id"]
    if "session_context" not in st.session_state:
        start_conversation(reader_id)

    with st.sidebar:
        st.markdown('<div class="sidebar-logo">📚 智阅通</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-note">当前读者：**{user["display_name"] or user["username"]}**'
            f'（ID：{reader_id}）</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # 模型切换：全部经百炼的 OpenAI 兼容端点调用，通义与第三方模型可混选
        model_id = model_selectbox()
        st.divider()

        st.markdown("**⚡ 快速提问**")
        for example in EXAMPLES:
            if st.button(example, width="stretch"):
                st.session_state["_pending_prompt"] = example

        st.divider()
        if st.button("🗑️ 清空对话", type="primary", width="stretch"):
            start_conversation(reader_id)   # 换月份重新开始，读者身份不变
            st.rerun()

        # 清空对话保留跨会话记忆（只重置消息与月份），若要连带忘掉记忆则点这里
        if st.button("🧠 清除长期记忆", width="stretch"):
            repo.save_memory_summary(reader_id, "")
            st.session_state["memory_summary"] = ""
            st.success("已清除该读者的跨会话记忆")

        st.divider()
        # 长期记忆与摘要：普通读者只能看到**自己账号下**的记忆，
        # 这里展示的就是当前登录读者（reader_id）自己的跨会话记忆摘要。
        with st.expander("🧠 长期记忆与摘要", expanded=True):
            memory_text = st.session_state.get("memory_summary", "")
            if memory_text:
                st.markdown("**记忆摘要**")
                st.write(memory_text)
            else:
                st.info("暂无长期记忆。对话超出 token 预算后，被压缩的历史会自动生成记忆摘要。")

        # 最近一轮的质检结果。
        # 注意：侧边栏在这里就渲染完了，而本轮的提问在下方 if prompt: 分支里才处理，
        # 因此本轮结束时必须 st.rerun() 一次，否则这个面板永远慢一轮。
        last_eval = st.session_state.get("last_eval")
        if last_eval:
            render_quality_panel(last_eval, expanded=False)
        else:
            st.caption("✅ 本轮质检：本轮提问后显示")

        with st.expander("ℹ️ 关于本项目"):
            st.markdown(
                "**技术栈**：LangChain · LangGraph · ReAct · Chroma · DashScope · MySQL\n\n"
                "支持**多轮对话上下文**，可追问「它」「那本书」「再推荐几本」。\n\n"
                "推荐书目均来自 MySQL 馆藏库，**只推荐在架且可借数量大于 0 的图书**。"
            )
            st.caption(f"本次对话读者ID：{st.session_state['session_context']['user_id']}")
            st.caption(f"当前对话模型：{model_id}")

        logout_button()

    header()

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # 助手消息上挂着该轮的质检结果，重放时一并显示。
            # 这个 "eval" 键不会流进模型：agent/memory.py 的 _clean_history
            # 只取 role/content 两个字段。
            if message.get("eval"):
                render_quality_panel(message["eval"])

    prompt = st.chat_input("请输入您的问题，例如：图书馆有哪些功能分区？")

    # 快捷提问触发后，优先处理侧边栏选中的问题
    pending = st.session_state.pop("_pending_prompt", None)
    if pending:
        prompt = pending

    if prompt:
        # 先取「本轮提问之前」的历史对话，再追加当前提问；
        # 否则当前提问会同时出现在历史与本轮输入中，被送进模型两次。
        history = list(st.session_state["messages"])

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # model_id 来自侧边栏下拉框；按模型缓存，切换后拿到的是对应模型的 Agent
        agent = get_agent(model_id)

        # ① 组装输入：按 token 预算裁剪历史，溢出部分压缩进记忆摘要。
        #    这一步可能触发一次额外的模型调用，因此放在流式渲染之前同步完成。
        prepared = agent.prepare(prompt, history, st.session_state.get("memory_summary", ""))

        # ② 摘要只在真正被更新时落库：项目是逐次建连、无连接池，不该每轮都写一次
        if prepared["changed"]:
            st.session_state["memory_summary"] = prepared["summary"]
            repo.save_memory_summary(reader_id, prepared["summary"])

        # 本轮的工具调用轨迹：中间件会往里追加，供质检分析。
        # 传的是一个普通 list，经 context 传给图执行，同一次 stream() 内共享同一对象。
        trace: list[dict] = []
        timing: dict = {"ttfb_ms": None, "latency_ms": None}

        # 助手消息（token 级流式输出，write_stream 返回完整文本）
        with st.chat_message("assistant"):
            try:
                response = st.write_stream(
                    timed_stream(
                        agent.execute_stream(
                            prepared["messages"],
                            session_context=st.session_state["session_context"],
                            memory_summary=prepared["summary"],
                            trace=trace,
                        ),
                        timing,
                    )
                )
            except Exception as e:
                # 模型调用失败时给出可读提示而非整页堆栈。鉴权类报错（401/403）
                # 交给 describe_dashscope_error 区分成因，避免把「Key 有效但权限受限」
                # 误判成「Key 无效」而带偏排查方向。
                logger.error(f"[对话]模型调用失败（model={model_id}）：{e}")
                st.error(_friendly_error(e, model_id))
                st.caption(f"详细信息：{e}")
                response = None
                # 崩溃轮也要留痕：工具抛异常会中断整轮执行，若不补记录，这一轮
                # 就从质检表里彻底消失，管理员看到的通过率会随着系统变坏而变高。
                quality_store.safe_record_failure(
                    query=prompt, error=str(e), reader_id=reader_id,
                    model_id=model_id, latency_ms=timing.get("latency_ms"),
                )

        if response:
            # ① 质检：纯旁路，失败不影响对话（store 里每个入口都吞异常）
            report = quality_store.safe_evaluate(
                query=prompt, response=response, trace=trace,
                reader_id=reader_id, model_id=model_id,
                latency_ms=timing.get("latency_ms"),
            )

            message = {"role": "assistant", "content": response}
            if report is not None:
                compact = report.to_compact()
                message["eval"] = compact
                st.session_state["last_eval"] = compact

            st.session_state["messages"].append(message)

            # ② 渲染本轮质检面板（挂在回答正下方）
            if report is not None:
                render_quality_panel(report.to_compact())

            # ③ 侧边栏在本轮处理之前就已渲染，必须重跑一次才能让它显示本轮结果。
            #    这是安全的：rerun 时 st.chat_input 返回 None，且 _pending_prompt
            #    已在上面被 pop 掉，因此 if prompt: 分支会被跳过 —— 不会二次调用模型、
            #    不会重复消耗 token，回答从 session_state 重放。
            st.rerun()


# ===============================================================
# 视图三：管理员后台
# ===============================================================
def _tab_users() -> None:
    """用户管理：查看 / 新增 / 启用禁用 / 重置密码 / 删除。"""
    users = repo.list_users()
    st.dataframe(users, width="stretch", hide_index=True)
    st.caption(f"共 {len(users)} 个账号")

    with st.expander("➕ 新增读者账号"):
        with st.form("admin_create_user"):
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("用户名")
                display_name = st.text_input("昵称")
            with col2:
                password = st.text_input("初始密码", type="password")
                reader_id = st.text_input("读者ID", placeholder="选填，默认同用户名")
            if st.form_submit_button("创建", type="primary"):
                if not username.strip() or not password:
                    st.error("用户名与密码不能为空")
                elif repo.get_user_by_username(username.strip()) is not None:
                    st.error("该用户名已存在")
                else:
                    try:
                        repo.create_user(
                            username=username.strip(),
                            password=password,
                            role="reader",
                            display_name=display_name.strip() or username.strip(),
                            reader_id=reader_id.strip() or None,
                        )
                        st.success(f"已创建读者账号 {username.strip()}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"创建失败：{e}")

    st.divider()
    st.markdown("**🔧 账号操作**")

    manageable = [u for u in users if u["role"] != "admin"]
    if not manageable:
        st.info("暂无可管理的读者账号")
        return

    labels = {f'{u["username"]}（{u["display_name"] or "-"}｜ID {u["reader_id"]}）': u
              for u in manageable}
    choice = st.selectbox("选择账号", list(labels.keys()), key="admin_user_select")
    target = labels[choice]

    col1, col2, col3 = st.columns(3)

    with col1:
        is_active = target["status"] == "active"
        if st.button("🚫 禁用账号" if is_active else "✅ 启用账号", width="stretch"):
            repo.update_user_status(target["id"], "disabled" if is_active else "active")
            st.success("已更新账号状态")
            st.rerun()

    with col2:
        new_pwd = st.text_input("新密码", type="password", key="admin_new_pwd")
        if st.button("🔑 重置密码", width="stretch"):
            if len(new_pwd) < 6:
                st.error("密码长度至少 6 位")
            else:
                repo.reset_password(target["id"], new_pwd)
                st.success(f"已重置 {target['username']} 的密码")
                st.rerun()

    with col3:
        confirm = st.checkbox("确认删除", key="admin_confirm_delete")
        if st.button("🗑️ 删除账号", width="stretch"):
            if not confirm:
                st.error("请先勾选「确认删除」")
            else:
                repo.delete_user(target["id"])
                st.success(f"已删除账号 {target['username']}")
                st.rerun()


def _tab_borrow() -> None:
    """借阅查询：选读者 → 看其借阅历史明细。"""
    readers = repo.list_readers()
    if not readers:
        st.info("暂无读者账号")
        return

    labels = {f'{r["reader_id"]}（{r["display_name"] or r["username"]}）': r["reader_id"]
              for r in readers}
    choice = st.selectbox("选择读者", list(labels.keys()), key="borrow_reader_select")
    reader_id = labels[choice]

    monthly = repo.get_borrow_monthly_all(reader_id)
    history = repo.get_borrow_history(reader_id)

    if not monthly and not history:
        st.info(f"读者 {reader_id} 暂无借阅记录")
        return

    if monthly:
        st.markdown(f"**月度借阅汇总**（共 {len(monthly)} 个月）")
        st.dataframe(monthly, width="stretch", hide_index=True)

    if history:
        st.divider()
        st.markdown(f"**借阅明细**（共 {len(history)} 条）")
        st.dataframe(history, width="stretch", hide_index=True)


def _tab_books() -> None:
    """图书管理：查看馆藏与库存 / 新增图书 / 调整库存。"""
    books = repo.list_books()
    available = sum(1 for b in books if b["available_stock"] > 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("馆藏品种", len(books))
    col2.metric("当前可借品种", available)
    col3.metric("已借空品种", len(books) - available)

    st.dataframe(books, width="stretch", hide_index=True)

    with st.expander("➕ 新增图书"):
        with st.form("admin_add_book"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("书名")
                author = st.text_input("作者")
                category = st.text_input("类别", placeholder="如 科幻 / 历史 / 经济")
            with col2:
                call_number = st.text_input("索书号", placeholder="如 I247.5/1200")
                location = st.text_input("馆藏位置", placeholder="如 2楼文学借阅区")
                total = st.number_input("总复本数", min_value=1, value=3, step=1)

            if st.form_submit_button("新增", type="primary"):
                if not title.strip() or not category.strip():
                    st.error("书名与类别不能为空")
                else:
                    try:
                        repo.add_book(
                            title=title.strip(),
                            author=author.strip() or "佚名",
                            category=category.strip(),
                            call_number=call_number.strip() or "待编目",
                            location=location.strip() or "待分配",
                            total_stock=int(total),
                        )
                        st.success(f"已新增《{title.strip()}》")
                        st.rerun()
                    except Exception as e:
                        st.error(f"新增失败（书名可能已存在）：{e}")

    st.divider()
    st.markdown("**📦 调整库存**")

    book_labels = {f'《{b["title"]}》｜{b["category"]}｜现可借 {b["available_stock"]}/{b["total_stock"]}': b
                   for b in books}
    choice = st.selectbox("选择图书", list(book_labels.keys()), key="book_select")
    book = book_labels[choice]

    col1, col2 = st.columns(2)
    with col1:
        new_total = st.number_input("总复本数", min_value=0,
                                    value=int(book["total_stock"]), step=1)
    with col2:
        new_available = st.number_input("可借数量", min_value=0,
                                        value=int(book["available_stock"]), step=1)

    if st.button("💾 保存库存", type="primary"):
        if new_available > new_total:
            st.error("可借数量不能大于总复本数")
        else:
            repo.update_book_stock(book["id"], int(new_total), int(new_available))
            st.success(f"已更新《{book['title']}》的库存")
            st.rerun()


def _tab_memory() -> None:
    """记忆与摘要：管理员查看所有读者的长期记忆摘要。

    管理员拥有全局视角：先看总览表（谁有记忆、谁还没有、最后更新时间），
    再选中某个读者查看其记忆全文，必要时可一键清除该读者的记忆。
    """
    rows = repo.list_readers_with_memory()
    if not rows:
        st.info("暂无读者账号")
        return

    overview = [
        {
            "读者ID": r["reader_id"],
            "昵称": r["display_name"] or "-",
            "记忆状态": "有记忆" if r["summary"] else "暂无",
            # updated_at 是 datetime，与占位符 "-" 混列会被 Arrow 拒绝序列化
            # （报 ArrowTypeError），统一转成字符串再进表格。
            "最后更新": r["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
            if r["updated_at"] else "-",
        }
        for r in rows
    ]
    st.dataframe(overview, width="stretch", hide_index=True)
    st.caption(f"共 {len(rows)} 个读者账号")

    st.divider()
    st.markdown("**🔍 查看记忆详情**")

    labels = {f'{r["reader_id"]}（{r["display_name"] or "-"}）': r for r in rows}
    choice = st.selectbox("选择读者", list(labels.keys()), key="memory_reader_select")
    target = labels[choice]

    if not target["summary"]:
        st.info(f"读者 {target['reader_id']} 暂无长期记忆")
    else:
        st.markdown(f"**读者 {target['reader_id']} 的长期记忆摘要**")
        st.caption(f"最后更新：{target['updated_at']}")
        st.write(target["summary"])

    if st.button("🗑️ 清除该读者的记忆", type="primary"):
        repo.save_memory_summary(target["reader_id"], "")
        st.success(f"已清除读者 {target['reader_id']} 的长期记忆")
        st.rerun()


def _tab_eval() -> None:
    """质检看板：用图表展示读者真实对话累积的质检记录。

    数据源**只有**读者真实对话的在线质检结果，不是标准化测试集 —— 这一点必须写在
    图表旁边，否则模型之间的小样本差异会被当成结论。
    """
    # ---------------- 全局筛选：一行，作用于下方所有图表 ----------------
    left, mid, right = st.columns([1, 2, 2])
    with left:
        window = st.selectbox("时间范围", ["近 7 天", "近 30 天", "近 90 天", "全部"],
                              index=1, key="eval_window")
    with mid:
        model_options = quality_store.safe_list_models()
        picked_models = st.multiselect("模型（不选=全部）", model_options,
                                       key="eval_models")
    with right:
        scenario_options = {spec.key: spec.label
                            for spec in quality_scenario.SCENARIOS.values()}
        picked_scenarios = st.multiselect(
            "场景（不选=全部）",
            list(scenario_options),
            format_func=lambda key: scenario_options.get(key, key),
            key="eval_scenarios")

    since = _eval_since(window)
    rows = quality_store.safe_load_records(
        since=since, limit=5000,
        model_ids=picked_models or None, scenarios=picked_scenarios or None)

    if not rows:
        st.info("暂无质检记录。读者每完成一轮对话就会产生一条，聊几轮后再回来看。")
        st.caption(
            "看板的数据源只有**读者真实对话**的在线质检结果，不是标准化测试集；"
            "模型之间的小样本差异不足以支撑结论（样本少于 "
            f"{quality_aggregate.MIN_SAMPLES_FOR_RANKING} 轮的模型不参与对比图）。"
        )
        return

    frame = quality_aggregate.to_frame(rows)

    # ---------------- KPI 行：一个数就是一个数，不画图 ----------------
    kpi = quality_aggregate.kpi_summary(frame)
    cards = st.columns(6)
    cards[0].metric("质控轮次", kpi["total"])
    cards[1].metric("全项通过率", quality_render.rate(kpi["pass_rate"]),
                    help="跨轮累加 sum(通过项)/sum(判定项)，只统计判定过的检查项")
    cards[2].metric("推荐 F1 均值", quality_render.rate(kpi["rec_f1"]),
                    help="只对有推荐结果的轮次求均值")
    cards[3].metric("工具召回均值", quality_render.rate(kpi["tool_recall"]),
                    help="只衡量提示词明令的必需工具，不惩罚额外调用")
    cards[4].metric("异常中断", kpi["failed"], delta=None,
                    delta_color="off", help="对话未正常完成的轮次")
    cards[5].metric("场景未识别率", quality_render.rate(kpi["unknown_share"]),
                    help="规则健康度指标：偏高说明关键词表覆盖不足")

    # 分母必须写出来，否则「推荐 F1 均值」这个数没有可信度
    st.caption(
        f"推荐 F1 均值基于 **{kpi['rec_f1_n']}** 轮有推荐结果的对话"
        f"（另 **{kpi['evaluable'] - kpi['rec_f1_n']}** 轮不适用，不计入）；"
        f"工具召回均值基于 **{kpi['tool_recall_n']}** 轮有必需工具的对话。"
        f"「—」表示无有效样本。"
    )
    st.divider()

    # ---------------- 1. 检查项失败分布 ----------------
    st.markdown("**① 检查项失败分布**")
    st.caption("失败率 = 该检查项未通过的轮次 / 该项被判定的轮次。"
               "整体偏高且平坦 → 多半是提示词的问题；只有一项突出 → 那条规则的问题。")
    failure = quality_aggregate.check_failure_rates(frame)
    if failure.empty:
        st.info("当前筛选范围内没有判定过的检查项。")
    else:
        st.altair_chart(quality_charts.failure_rate_chart(failure), width="stretch")
        with st.expander("📋 表格视图"):
            st.dataframe(
                failure.assign(失败率=lambda d: (d["failure_rate"] * 100).round(1))
                       .rename(columns={"check_name": "检查项", "determined": "判定轮次",
                                        "failed": "失败轮次", "失败率": "失败率(%)"}),
                width="stretch", hide_index=True)
    st.divider()

    # ---------------- 2. 按模型对比 ----------------
    st.markdown("**② 按模型的工具召回率 vs 全项通过率**")
    st.caption("工具指标只衡量提示词明令的必需/禁用工具，不惩罚模型额外调用的"
               "合理解释性工具。样本不足的模型不参与排名。")
    enough, scarce = quality_aggregate.by_model(frame)
    if enough.empty:
        st.info(f"没有模型达到 {quality_aggregate.MIN_SAMPLES_FOR_RANKING} 轮样本，暂不做横向对比。")
    else:
        st.altair_chart(quality_charts.model_comparison_chart(enough), width="stretch")
        with st.expander("📋 表格视图"):
            st.dataframe(
                enough.assign(工具召回率=lambda d: (d["tool_recall"] * 100).round(1),
                              全项通过率=lambda d: (d["pass_rate"] * 100).round(1),
                              推荐F1=lambda d: (d["rec_f1"] * 100).round(1))
                      .rename(columns={"model_id": "模型", "n": "样本轮次"})[
                          ["模型", "样本轮次", "工具召回率", "全项通过率", "推荐F1"]],
                width="stretch", hide_index=True)
    if not scarce.empty:
        st.caption("样本不足，未参与对比："
                   + "；".join(f"{r.model_id}（{r.n} 轮）" for r in scarce.itertuples()))

    # ---------------- 3. 趋势：通过率 + 样本量（共享 x 轴的两张图）----------------
    st.divider()
    st.markdown("**③ 通过率与样本量的时间趋势**")
    st.caption("通过率必须和样本量一起看 —— 某天 1 轮全过和 40 轮全过都是 100%。"
               "两张图共用一条时间轴，而不是把两个量级画进同一张图的两个 y 轴。")
    trend = quality_aggregate.daily_trend(frame)
    if trend.empty:
        st.info("当前筛选范围内没有可统计的轮次。")
    else:
        st.altair_chart(quality_charts.pass_rate_trend_chart(trend), width="stretch")
        st.altair_chart(quality_charts.sample_size_chart(trend), width="stretch")
        with st.expander("📋 表格视图"):
            st.dataframe(
                trend.assign(通过率=lambda d: (d["pass_rate"] * 100).round(1))
                     .rename(columns={"date": "日期", "n": "轮次", "pass_rate": "通过率(%)"}),
                width="stretch", hide_index=True)
    st.divider()

    # ---------------- 4. 场景 × 检查项 热力图 ----------------
    st.markdown("**④ 场景 × 检查项 通过率**")
    st.caption("用来把「模型不擅长某类场景」和「该场景的必需工具规则定错了」分开："
               "若某个场景整列都在同一项上失败，先怀疑规则，而不是模型。")
    matrix = quality_aggregate.scenario_check_matrix(frame)
    if matrix.empty:
        st.info("当前筛选范围内没有判定过的检查项。")
    else:
        st.altair_chart(quality_charts.scenario_check_heatmap(matrix), width="stretch")
        with st.expander("📋 表格视图"):
            st.dataframe(
                matrix.assign(通过率=lambda d: (d["pass_rate"] * 100).round(1))
                      .rename(columns={"scenario_label": "场景", "check_name": "检查项",
                                       "n": "轮次", "pass_rate": "通过率(%)"}),
                width="stretch", hide_index=True)

    # ---------------- 5. 场景判定健康度 ----------------
    share = quality_aggregate.other_share(frame)
    st.divider()
    with st.expander("🔍 场景判定健康度（用于审计关键词规则本身）"):
        st.markdown(
            f"- **未识别场景（other）占比**：{quality_render.rate(share['other_share'])}"
            f"　—— 这些轮次的工具指标是「不适用」，不会造成错误扣分，但覆盖不到就发现不了问题\n"
            f"- **判定来源为 unknown 的比例**：{quality_render.rate(share['unknown_share'])}"
            f"　—— 与上一项同源，偏高说明关键词表需要补词\n"
            f"- 当前筛选范围内共 **{share['total']}** 轮"
        )
        st.caption("场景判定用一张刻意收窄的关键词表：模糊提问一律落到 other 并记为"
                   "「不适用」，因此**误判永远不会产生错误扣分**，只会少算覆盖。"
                   "这张表是否够用，靠上面这两个数来判断，而不是靠信任它。")

    st.divider()
    st.caption(
        "**数据口径**：以上全部来自读者真实对话的在线质检，不是标准化测试集；"
        "所有比率列的 NULL 表示「本轮不适用」（不是 0），聚合时会跳过。"
        "场景由关键词规则判定，仅供定位问题，不作为对模型的整体评分。"
    )


def _eval_since(window: str) -> str | None:
    """把时间范围选项换算成 ``YYYY-MM-DD HH:MM:SS``；「全部」返回 None。"""
    days = {"近 7 天": 7, "近 30 天": 30, "近 90 天": 90}.get(window)
    if days is None:
        return None
    # 用「今天 0 点往前推 N 天」而不是「此刻往前推 N×24 小时」，
    # 这样「近 7 天」就是完整的 7 个自然日，和读者看到日期时的直觉一致。
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) \
        - timedelta(days=days - 1)
    return start.strftime("%Y-%m-%d %H:%M:%S")


def render_admin(user: dict) -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">📚 智阅通 · 后台</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-note">管理员：**{user["display_name"] or user["username"]}**</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "**功能导航**\n\n"
            "- 用户管理：增删账号、启用/禁用、重置密码\n"
            "- 借阅查询：查看指定读者的借阅历史\n"
            "- 图书管理：维护馆藏与可借库存\n"
            "- 记忆与摘要：查看读者的长期记忆\n"
            "- 质检看板：对话质量指标的图表统计"
        )
        logout_button()

    header()
    st.markdown("### 🛠️ 管理后台")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["👥 用户管理", "📖 借阅查询", "📚 图书管理", "🧠 记忆与摘要", "📊 质检看板"]
    )
    with tab1:
        _tab_users()
    with tab2:
        _tab_borrow()
    with tab3:
        _tab_books()
    with tab4:
        _tab_memory()
    with tab5:
        _tab_eval()


# ===============================================================
# 分发
# ===============================================================
current_user = st.session_state.get("user")

if current_user is None:
    render_login()
elif current_user["role"] == "admin":
    render_admin(current_user)
else:
    render_chat(current_user)
