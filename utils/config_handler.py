
import os

import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path

# 在此加载 .env：config_handler 被所有模块导入，是唯一能保证
# 任何入口（app.py / python -m db.init_db / 单文件脚本）都拿到环境变量的地方。
# 这里强制 override=True，确保项目根目录的 .env 能覆盖任何 shell / IDE 中残留的旧值，
# 否则就可能一直拿到失效的 DASHSCOPE_API_KEY，造成 401 InvalidApiKey。
load_dotenv(override=True)


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


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # 容器化部署时不用改 YAML 就能换默认模型（同一镜像可跑不同模型）。
    # 只覆盖「默认选中的模型」：可选清单 chat_model_presets 仍以 YAML 为准——
    # 列表型配置用环境变量表达易错，且运行期在侧边栏切换已足够覆盖容器场景。
    return _apply_env_overrides(conf, {
        "CHAT_MODEL_NAME": ("chat_model_name", str),
    })


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # 向量库与 MD5 台账必须同进同退，容器里统一收敛到同一个数据卷下，
    # 避免「卷里留了 chroma 却丢了 md5」导致重复入库。
    return _apply_env_overrides(conf, {
        "CHROMA_PERSIST_DIR": ("persist_directory", str),
        "CHROMA_MD5_STORE": ("md5_hex_store", str),
        "CHROMA_DATA_PATH": ("data_path", str),
    })


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_db_config(config_path: str = get_abs_path("config/db.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # 容器内 db.yml 里的 host=127.0.0.1 指向的是容器自己，必须覆盖为 compose 的服务名。
    # 口令仍只从 .env 的 MYSQL_PASSWORD 读，不经这里。
    return _apply_env_overrides(conf, {
        "MYSQL_HOST": ("host", str),
        "MYSQL_PORT": ("port", int),
        "MYSQL_USER": ("user", str),
        "MYSQL_DATABASE": ("database", str),
    })


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
agent_conf = load_agent_config()
prompts_conf = load_prompts_config()
db_conf = load_db_config()
