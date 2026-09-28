"""模型工厂：统一创建对话模型与向量化模型。

**为什么对话模型走 OpenAI 兼容端点而不是 ChatTongyi**

百炼上架的模型分两类：通义系列，以及第三方模型（DeepSeek / GLM / Llama 等）。
`langchain_community` 的 `ChatTongyi` 封装的是 DashScope 原生协议，只能调通通义
系列；第三方模型必须走百炼的 **OpenAI 兼容端点**（`compatible-mode`）。与其维护
两套调用路径，不如统一收敛到兼容端点——它对通义系列同样完全支持，于是「换模型」
就退化成「换一个 model 字符串」，无需改代码。

顺带解决了两件事：
- `ChatOpenAI` 原生支持流式 + 工具调用的组合，原先为 `ChatTongyi` 打的
  `incremental_output` 补丁（见 git 历史）不再需要；
- 模型清单可以在 config/rag.yml 里声明，前端据此渲染下拉框。

向量化则**不能**跟着走兼容端点——百炼的 embedding 只有通义 `text-embedding-*`
系列，且未提供 OpenAI 兼容接口，因此仍由 dashscope SDK 直接调用。
"""
import os
from typing import Any, Dict, List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

# .env 的加载统一由 utils.config_handler 负责（导入它即触发 load_dotenv），
# 此处不再重复调用，避免两处各留一份、日后改了一处漏另一处。
from utils.config_handler import rag_conf

# 百炼的 OpenAI 兼容端点（华北2·北京）。若使用新加坡地域的百炼，改为
# https://dashscope-intl.aliyuncs.com/compatible-mode/v1
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# config/rag.yml 未配置或配置为空时的兜底对话模型
DEFAULT_CHAT_MODEL = "qwen3-max"

# 视为「尚未填写」的占位值，避免把模板里的示例串当成真实 Key 用
_PLACEHOLDER_KEY_VALUES = {
    "your_dashscope_api_key_here",
    "your_api_key_here",
    "changeme",
    "replace_me",
    "sk-xxxxxx",
    "sk-在此填入你的百炼api key",
}


def _require_dashscope_key() -> str:
    """校验 DashScope Key 是否真实可用，避免继续使用旧失效值。"""
    candidates = ("DASHSCOPE_API_KEY", "QWEN_API_KEY")
    for env_name in candidates:
        value = os.getenv(env_name, "").strip()
        if value and value.lower() not in _PLACEHOLDER_KEY_VALUES:
            os.environ["DASHSCOPE_API_KEY"] = value
            return value

    raise ValueError(
        "DashScope API Key 未配置或仍为占位值。请在项目根目录的 .env 中设置 "
        "DASHSCOPE_API_KEY=...（申请地址：https://bailian.console.aliyun.com/）"
    )


_require_dashscope_key()


def build_chat_model(model_id: str) -> ChatOpenAI:
    """按模型 ID 构造对话模型，指向百炼的 OpenAI 兼容端点。

    不缓存实例：调用方（app.py 的 ``st.cache_resource``）已按 model_id 做了缓存，
    工厂只管构造，避免两处各留一份缓存、失效时机对不上。

    :param model_id: 百炼模型广场上的模型 ID，如 ``qwen3.8-max`` / ``deepseek-v4-pro``。
                     模型必须已在百炼控制台开通，否则调用时会被鉴权层拒绝。
    """
    model_id = (model_id or "").strip() or DEFAULT_CHAT_MODEL

    return ChatOpenAI(
        model=model_id,
        # 必须显式传入：ChatOpenAI 默认读 OPENAI_API_KEY，而本项目用的是百炼的 Key
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=DASHSCOPE_COMPATIBLE_BASE_URL,
        # 与原先 ChatTongyi 的配置保持一致：开启流式，前端才能逐字输出
        streaming=True,
        # 不设 temperature/top_p，沿用兼容端点的服务端默认值，
        # 与改造前 ChatTongyi 的行为对齐
    )


def list_chat_model_presets() -> List[Dict[str, str]]:
    """读取 config/rag.yml 中声明的可选模型清单，供前端渲染下拉框。

    对配置做规范化：缺 id 的条目丢弃，缺 label 的用 id 兜底。
    清单为空时退回「仅含默认模型」的单元素列表，保证下拉框永远有可选项。
    """
    presets: List[Dict[str, str]] = []

    for item in rag_conf.get("chat_model_presets") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        presets.append({
            "label": str(item.get("label") or model_id).strip(),
            "id": model_id,
        })

    if not presets:
        fallback = str(rag_conf.get("chat_model_name") or DEFAULT_CHAT_MODEL).strip()
        presets = [{"label": fallback, "id": fallback}]

    return presets


def default_chat_model_id() -> str:
    """默认选中的模型：取配置的 chat_model_name，不在清单内则退回清单第一项。"""
    configured = str(rag_conf.get("chat_model_name") or "").strip()
    presets = list_chat_model_presets()

    if any(p["id"] == configured for p in presets):
        return configured

    return presets[0]["id"]


def describe_dashscope_error(exc: Exception, model_id: str | None = None) -> str | None:
    """把百炼的鉴权/权限报错翻译成可直接照做的排查提示。

    401 与 403 在 Streamlit 里长得几乎一样（都是一坨 HTTP error occurred…），
    成因却完全不同：401 是 Key 本身无效，403 是 Key 有效但被权限配置挡住。
    混成一句「请检查 API Key 是否有效」会把排查方向带偏——本次 403 就是如此。

    改造为 OpenAI 兼容端点后，异常类型变成了 openai 的 AuthenticationError /
    PermissionDeniedError，但**响应体里的错误码没变**（仍是 invalid_api_key /
    AccessDenied），因此继续按错误码字符串匹配，比按异常类型匹配更稳——
    换个 SDK 或 SDK 大版本升级都不至于让提示失效。

    :param model_id: 当前使用的模型，用于让提示指向具体模型。为空则取默认模型。
    :return: 已知的鉴权类错误返回中文提示；否则返回 None，由调用方按通用错误处理。
    """
    text = str(exc)
    lowered = text.lower()
    active_model = model_id or default_chat_model_id()

    if "AccessDenied" in text:
        return (
            f"API Key 本身有效（能通过鉴权），但被百炼的「API-Key 权限限制」拒绝了。"
            f"请到百炼控制台 → API-KEY 管理 → 编辑该 Key："
            f"① 权限改为「全部」，或确认「可访问模型」里已勾选 {active_model}"
            f"（以及向量化模型 {rag_conf['embedding_model_name']}）；"
            "② 若配置了 IP 白名单，确认当前公网出口 IP（含代理出口 IP）在白名单内。"
        )

    if "InvalidApiKey" in text or "invalid_api_key" in text:
        return (
            "API Key 无效或已失效。请在百炼控制台重新生成，并更新项目根目录 .env 的 "
            "DASHSCOPE_API_KEY；同时留意系统/IDE 环境变量里是否残留了旧 Key。"
        )

    if "Arrearage" in text:
        return "百炼账号已欠费，请充值后重试。"

    # 免费额度用尽：账号若开着「仅使用免费额度」，没有免费额度的模型会直接 403。
    # 这条与「欠费」不是一回事——账号可能还有余额，只是被这个模式挡住了。
    if "FreeTierOnly" in text or "Free quota exhausted" in text:
        return (
            f"模型 {active_model} 的免费额度已用尽。两种解法：① 到百炼控制台关闭"
            f"「仅使用免费额度」模式（需账号有余额），即可按量付费继续调用；"
            f"② 或在侧边栏换用仍带免费额度的模型（如 qwen-plus / deepseek-v3.2）。"
        )

    # 模型不存在 / 未开通：下拉框里挑了一个控制台没开通的模型时会走到这里。
    # 兼容端点返回 404，报文是 "The model `X` does not exist or you do not have access to it"。
    if ("ModelNotExist" in text or "model not found" in lowered or "model_not_found" in lowered
            or "does not exist or you do not have access" in lowered):
        return (
            f"模型 {active_model} 不存在或尚未开通。请到百炼控制台「模型广场」开通该模型，"
            f"或在 config/rag.yml 的 chat_model_presets 中改成一个已开通的模型 ID"
            f"（可用 ID 可查 https://dashscope.aliyuncs.com/compatible-mode/v1/models ）。"
        )

    # 模型不支持工具调用：推理类模型（如 DeepSeek-R1）常见。
    # 本项目的 ReAct Agent 依赖 function calling 检索馆藏与借阅数据，缺了就完全跑不动，
    # 因此这里给出明确结论，而不是让读者看到一坨 400 报错。
    if "tool" in lowered and ("not support" in lowered or "unsupported" in lowered):
        return (
            f"模型 {active_model} 不支持工具调用（function calling），而本项目的 ReAct Agent "
            "依赖它来检索馆藏与借阅数据，因此无法使用。请在侧边栏换一个支持工具调用的模型"
            "（通义系列、DeepSeek-V3 支持；DeepSeek-R1 等推理模型通常不支持）。"
        )

    return None


# 向量化模型：不随对话模型切换，因此仍在模块导入期构造一次即可
embed_model = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
