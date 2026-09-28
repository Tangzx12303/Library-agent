import os

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# 提示词缓存：{绝对路径: (文件修改时间, 内容)}
# 命中缓存时直接返回内存内容，避免 dynamic_prompt 中间件每次调用模型都读一次磁盘；
# 文件被改动后 mtime 变化，缓存自动失效，因此仍然支持运行时热改提示词。
_prompt_cache: dict[str, tuple[float, str]] = {}


def _load_prompt(config_key: str, error_tag: str) -> str:
    """按 config/prompts.yml 中的键读取提示词文件内容（带 mtime 缓存）。"""
    try:
        prompt_path = get_abs_path(prompts_conf[config_key])
    except KeyError as e:
        logger.error(f"[{error_tag}]在yaml配置项中没有{config_key}配置项")
        raise e

    try:
        mtime = os.path.getmtime(prompt_path)

        cached = _prompt_cache.get(prompt_path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        _prompt_cache[prompt_path] = (mtime, content)
        return content
    except Exception as e:
        logger.error(f"[{error_tag}]解析提示词出错，{str(e)}")
        raise e


def load_system_prompts() -> str:
    return _load_prompt("main_prompt_path", "load_system_prompts")


def load_rag_prompts() -> str:
    return _load_prompt("rag_summarize_prompt_path", "load_rag_prompts")


def load_report_prompts() -> str:
    return _load_prompt("report_prompt_path", "load_report_prompts")


def load_memory_summary_prompts() -> str:
    return _load_prompt("memory_summary_path", "load_memory_summary_prompts")
