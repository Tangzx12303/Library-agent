"""pytest 全局夹具：把有导入期副作用的模块替换成桩。

**本文件顶部的桩必须跑在任何业务模块被导入之前**，因此这里不能出现
任何 ``from agent... / from db... / from quality...`` 形式的导入。

为什么不能靠「先设环境变量」来绕过 Key 校验
--------------------------------------------------
直觉做法是在这里 ``os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")``，
但它**解决不了问题**：``utils/config_handler.py`` 在模块导入期就调用
``load_dotenv(override=True)``，而 .env 里那份值会**覆盖**掉我们刚设的值。
于是当某台机器的 .env 还停留在 ``sk-在此填入你的百炼API Key``（这个值恰好
在 ``model/factory.py`` 的占位符黑名单里）时，``_require_dashscope_key()``
会抛 ValueError，**整个测试套件在收集阶段就崩**。

本机 .env 存的是真实 Key，所以恰好能跑——这是最坏的一类 bug：本地看不见，
CI 上必炸。改环境变量治不了，直接把模块换掉才行。

已知并接受的副作用
--------------------------------------------------
``utils/logger_handler`` 在导入时会 ``os.makedirs(logs/)`` 并打开当天的日志
文件。这个副作用无害（logs/ 已 gitignore），且要挡掉它必须在**第一次导入之前**
做猴补丁，脆弱且收益为零，因此**有意保留**。日志里出现的 pytest 运行记录属正常。
"""
import sys
import types
from pathlib import Path

# ----------------------------------------------------------------------
# 0. 把项目根塞进 sys.path
#    pytest 在「无 __init__.py」的布局下只会把 tests/ 加进 sys.path，
#    用 `pytest`（而非 `python -m pytest`）启动时 import agent/db/quality 会失败。
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# 1. 桩掉 model.factory
#    它在导入期就执行 _require_dashscope_key()（会抛错）并构造 DashScopeEmbeddings。
#    DashScopeEmbeddings 的构造函数本身不联网，但我们不想让测试依赖 .env 里
#    到底填没填 Key，所以整个模块换掉。
# ----------------------------------------------------------------------
def _stub(name: str, **attrs) -> None:
    """把 name 注册成一个只带给定属性的空模块（已存在则不覆盖）。"""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub(
    "model.factory",
    embed_model=object(),
    build_chat_model=lambda model_id: None,
    list_chat_model_presets=lambda: [{"label": "stub", "id": "stub-model"}],
    default_chat_model_id=lambda: "stub-model",
    describe_dashscope_error=lambda exc, model_id=None: None,
)


# ----------------------------------------------------------------------
# 2. 桩掉 rag.rag_service
#    agent_tools 只用到它的 RagSummarizeService 这一个类；挡掉它同时省下
#    rag.vector_store → langchain_chroma → chromadb 的整条导入（秒级）。
# ----------------------------------------------------------------------
class _StubRagSummarizeService:
    """不做检索的替身：只记录拿到的模型，返回固定串。"""

    def __init__(self, model=None):
        self.model = model

    def rag_summarize(self, query: str) -> str:
        return f"[stub rag] {query}"


_stub("rag.rag_service", RagSummarizeService=_StubRagSummarizeService)


# ----------------------------------------------------------------------
# 3. 常用夹具
# ----------------------------------------------------------------------
import pytest  # noqa: E402  （必须在装桩之后导入）


@pytest.fixture
def no_db(monkeypatch):
    """把所有数据库出口换成会抛异常的替身。

    用于「这个测试绝不该碰数据库」的场景：打的是 ``db.repository`` 上的名字，
    因为 repository 是 ``from db.connection import query_all`` 导入的——
    打 ``db.connection.query_all`` 对它**毫无效果**，那会让测试在"通过"的
    同时真的连上数据库。
    """
    def _boom(*args, **kwargs):
        raise AssertionError(
            "本测试不应访问数据库；若确实需要，请显式 monkeypatch 对应的仓库函数"
        )

    import db.repository as repo

    for name in ("query_all", "query_one", "execute"):
        monkeypatch.setattr(repo, name, _boom)
    return repo


@pytest.fixture
def bound_reader():
    """把会话上下文绑定到 reader1001，测试结束后不残留。

    返回一个 setter：``bound_reader("1001")``。
    """
    from agent.tools import agent_tools

    def _bind(reader_id, month="2025-04"):
        agent_tools.apply_session_context({"user_id": reader_id, "month": month})
        return reader_id

    yield _bind

    # 复位，避免跨测试串味
    agent_tools._session_user_id.set(None)
    agent_tools._session_month.set(None)
