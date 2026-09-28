"""建库建表 + 灌入种子数据。用法：``python -m db.init_db``

脚本是**幂等**的，可以反复执行：
  - 建库建表用 IF NOT EXISTS；
  - 用户/图书用 INSERT IGNORE，已存在的不覆盖（保留管理员的后续改动）；
  - 借阅数据仅在表为空时灌入。

数据来源：
  - 图书目录：本文件内的 SEED_BOOKS（扩充后的馆藏书目，含可借库存）；
  - 借阅汇总：迁移自 data/external/records.csv（迁移后 CSV 不再被运行时读取）。
"""
import csv
import os
from datetime import date, timedelta

from db.connection import get_connection, get_server_connection, execute, query_one
from utils.config_handler import db_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.security import hash_password

# ----------------------------------------------------------------------
# 建表 DDL
# ----------------------------------------------------------------------
DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('reader','admin') NOT NULL DEFAULT 'reader',
    display_name  VARCHAR(64)  DEFAULT NULL,
    reader_id     VARCHAR(16)  NOT NULL UNIQUE,
    status        ENUM('active','disabled') NOT NULL DEFAULT 'active',
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_BOOKS = """
CREATE TABLE IF NOT EXISTS books (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    title           VARCHAR(128) NOT NULL UNIQUE,
    author          VARCHAR(128) DEFAULT NULL,
    category        VARCHAR(64)  NOT NULL,
    call_number     VARCHAR(32)  NOT NULL,
    location        VARCHAR(64)  NOT NULL,
    total_stock     INT NOT NULL DEFAULT 1,
    available_stock INT NOT NULL DEFAULT 1,
    status          VARCHAR(32)  DEFAULT '在架可借',
    KEY idx_category (category),
    KEY idx_available (available_stock)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_BORROW_RECORDS = """
CREATE TABLE IF NOT EXISTS borrow_records (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    reader_id   VARCHAR(16)  NOT NULL,
    book_title  VARCHAR(128) NOT NULL,
    month       CHAR(7)      NOT NULL,
    borrow_date DATE         DEFAULT NULL,
    due_date    DATE         DEFAULT NULL,
    return_date DATE         DEFAULT NULL,
    status      ENUM('borrowed','returned','overdue') NOT NULL DEFAULT 'borrowed',
    KEY idx_reader (reader_id),
    KEY idx_reader_month (reader_id, month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_BORROW_MONTHLY = """
CREATE TABLE IF NOT EXISTS borrow_monthly (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    reader_id     VARCHAR(16)  NOT NULL,
    month         CHAR(7)      NOT NULL,
    preference    VARCHAR(128) NOT NULL,
    borrow_count  INT NOT NULL,
    reading_hours INT NOT NULL,
    current_books VARCHAR(256) NOT NULL,
    comparison    VARCHAR(64)  NOT NULL,
    UNIQUE KEY uq_reader_month (reader_id, month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 对话记忆：溢出历史的压缩摘要，按读者存一行。
# 这是本项目唯一的「跨会话」记忆——多轮对话里被 token 预算挤出去的内容会压进
# 这份摘要，读者下次登录（甚至换一台设备）仍能续上，而不是从零开始。
DDL_CONVERSATION_MEMORY = """
CREATE TABLE IF NOT EXISTS conversation_memory (
    reader_id   VARCHAR(16) PRIMARY KEY,
    summary     TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# 对话质检记录：每轮对话一行，供读者看本轮结果、管理员看历史图表。
#
# 几个列名/类型上的讲究：
#   * rec_precision 而非 precision —— 避开 MySQL 关键字歧义；
#   * 比率列一律 DOUBLE DEFAULT NULL，不用 DECIMAL：pymysql 会把 DECIMAL 返回成
#     Decimal 对象，进 st.dataframe 时 Arrow 不认（项目已栽过一次 ArrowTypeError）；
#   * 比率列**必须可为 NULL**：不适用 ≠ 0。MySQL 的 AVG() 自动跳过 NULL，所以
#     知识问答这类没有推荐的轮次不会把推荐 F1 的均值往下拽；
#   * detail_json 用 MEDIUMTEXT 而非 JSON —— pymysql 对 JSON 列本来就返回 str，
#     用文本 + 显式 json.dumps/loads 少一个变量。**只存派生事实**，不存工具原始返回。
DDL_EVAL_TURN_RECORDS = """
CREATE TABLE IF NOT EXISTS eval_turn_records (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    reader_id       VARCHAR(16)  NOT NULL,
    model_id        VARCHAR(64)  NOT NULL,
    query           VARCHAR(512) NOT NULL,
    scenario        VARCHAR(16)  NOT NULL,
    scenario_source VARCHAR(16)  NOT NULL,
    passed          INT NOT NULL,
    determined      INT NOT NULL,
    rec_precision   DOUBLE DEFAULT NULL,
    rec_recall      DOUBLE DEFAULT NULL,
    rec_f1          DOUBLE DEFAULT NULL,
    tool_precision  DOUBLE DEFAULT NULL,
    tool_recall     DOUBLE DEFAULT NULL,
    tool_f1         DOUBLE DEFAULT NULL,
    latency_ms      INT DEFAULT NULL,
    detail_json     MEDIUMTEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_created (created_at),
    KEY idx_model_time (model_id, created_at),
    KEY idx_reader (reader_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# ----------------------------------------------------------------------
# 种子书目：65 本，覆盖 19 个类别
# 字段：书名, 作者, 类别, 索书号, 馆藏位置, 总复本, 可借数量
#
# 其中若干本 available_stock = 0（已借空）：既是真实场景，也用于验证
# 「推荐图书必须有可借库存」这条约束确实生效——它们不应出现在推荐里。
# ----------------------------------------------------------------------
SEED_BOOKS = [
    # ---- 科幻 ----
    ("三体", "刘慈欣", "科幻", "I247.5/1200", "2楼文学借阅区", 5, 3),
    ("流浪地球", "刘慈欣", "科幻", "I247.5/1201", "2楼文学借阅区", 4, 4),
    ("基地", "艾萨克·阿西莫夫", "科幻", "I712.45/0331", "2楼文学借阅区", 3, 2),
    ("沙丘", "弗兰克·赫伯特", "科幻", "I712.45/0332", "2楼文学借阅区", 3, 0),
    ("银河系漫游指南", "道格拉斯·亚当斯", "科幻", "I561.45/0218", "2楼文学借阅区", 3, 1),
    # ---- 历史 ----
    ("万历十五年", "黄仁宇", "历史", "K248.3/016", "3楼历史地理阅览区", 4, 2),
    ("史记", "司马迁", "历史", "K204.2/005", "3楼历史地理阅览区", 3, 1),
    ("人类简史", "尤瓦尔·赫拉利", "历史", "K02-49/045", "3楼历史地理阅览区", 4, 0),
    ("枪炮、病菌与钢铁", "贾雷德·戴蒙德", "历史", "K02/0072", "3楼历史地理阅览区", 3, 2),
    # ---- 哲学 ----
    ("中国哲学简史", "冯友兰", "哲学", "B2/0091", "3楼社会科学阅览区", 4, 3),
    ("苏菲的世界", "乔斯坦·贾德", "哲学", "B1-49/0022", "3楼社会科学阅览区", 5, 4),
    ("沉思录", "马可·奥勒留", "哲学", "B502.43/011", "3楼社会科学阅览区", 3, 2),
    ("理想国", "柏拉图", "哲学", "B502.232/006", "3楼社会科学阅览区", 3, 1),
    # ---- 计算机 ----
    ("代码大全", "史蒂夫·麦康奈尔", "计算机", "TP311.52/0140", "4楼自然科学阅览区", 3, 0),
    ("深入理解计算机系统", "兰德尔·布莱恩特", "计算机", "TP303/0055", "4楼自然科学阅览区", 2, 2),
    ("算法导论", "托马斯·科尔曼", "计算机", "TP301.6/0038", "4楼自然科学阅览区", 3, 1),
    ("流畅的Python", "卢西亚诺·拉马略", "计算机", "TP311.561/0121", "4楼自然科学阅览区", 3, 3),
    # ---- 科普 ----
    ("时间简史", "史蒂芬·霍金", "科普", "P159-49/032", "4楼自然科学阅览区", 5, 3),
    ("上帝掷骰子吗", "曹天元", "科普", "O413-49/0017", "4楼自然科学阅览区", 4, 2),
    ("自私的基因", "理查德·道金斯", "科普", "Q111-49/0009", "4楼自然科学阅览区", 3, 2),
    ("万物简史", "比尔·布莱森", "科普", "N49/0218", "4楼自然科学阅览区", 4, 4),
    # ---- 经济 ----
    ("经济学原理", "曼昆", "经济", "F0/0110", "3楼社会科学阅览区", 5, 2),
    ("国富论", "亚当·斯密", "经济", "F091.33/0024", "3楼社会科学阅览区", 3, 2),
    ("置身事内", "兰小欢", "经济", "F124/0087", "3楼社会科学阅览区", 4, 3),
    ("穷查理宝典", "彼得·考夫曼", "经济", "F830.59/0233", "3楼社会科学阅览区", 4, 1),
    # ---- 管理 ----
    ("原则", "瑞·达利欧", "管理", "C93/0156", "3楼社会科学阅览区", 4, 2),
    ("从优秀到卓越", "吉姆·柯林斯", "管理", "F272.91/0098", "3楼社会科学阅览区", 3, 2),
    ("卓有成效的管理者", "彼得·德鲁克", "管理", "C93/0157", "3楼社会科学阅览区", 3, 3),
    # ---- 绘画 ----
    ("艺术的故事", "贡布里希", "绘画", "J110.9/0042", "3楼社会科学阅览区", 4, 0),
    ("梵高手稿", "文森特·梵高", "绘画", "J231/0013", "3楼社会科学阅览区", 3, 0),
    ("西方美术史", "张敢", "绘画", "J110.9/0043", "3楼社会科学阅览区", 4, 3),
    # ---- 音乐 ----
    ("西方音乐史", "保罗·亨利·朗", "音乐", "J609.5/0008", "3楼社会科学阅览区", 3, 1),
    ("聆听音乐", "克雷格·莱特", "音乐", "J605/0021", "3楼社会科学阅览区", 4, 3),
    ("音乐是怎样算成的", "伊莱·马奥尔", "音乐", "J60-05/0004", "3楼社会科学阅览区", 3, 2),
    # ---- 绘本 ----
    ("猜猜我有多爱你", "山姆·麦克布雷尼", "绘本", "I287.8/210", "1楼少儿阅览区", 6, 4),
    ("窗边的小豆豆", "黑柳彻子", "绘本", "I313.86/0027", "1楼少儿阅览区", 5, 3),
    ("野兽国", "莫里斯·桑达克", "绘本", "I712.85/0114", "1楼少儿阅览区", 4, 2),
    ("逃家小兔", "玛格丽特·怀兹·布朗", "绘本", "I712.85/0115", "1楼少儿阅览区", 4, 3),
    # ---- 教育 ----
    ("好妈妈胜过好老师", "尹建莉", "教育", "G78/0233", "1楼少儿阅览区", 5, 2),
    ("正面管教", "简·尼尔森", "教育", "G78/0234", "1楼少儿阅览区", 4, 3),
    ("孩子：挑战", "鲁道夫·德雷克斯", "教育", "G78/0235", "1楼少儿阅览区", 3, 2),
    ("爱和自由", "孙瑞雪", "教育", "G61/0067", "1楼少儿阅览区", 3, 1),
    # ---- 养生 ----
    ("黄帝内经", "佚名", "养生", "R221/018", "4楼自然科学阅览区", 4, 2),
    ("睡眠革命", "尼克·利特尔黑尔斯", "养生", "R338.63/0011", "4楼自然科学阅览区", 4, 3),
    ("营养学", "葛可佑", "养生", "R151/0045", "4楼自然科学阅览区", 3, 2),
    # ---- 医学 ----
    ("人体简史", "比尔·布莱森", "医学", "R32-49/0019", "4楼自然科学阅览区", 4, 3),
    ("医学的温度", "韩启德", "医学", "R-49/0006", "4楼自然科学阅览区", 3, 2),
    ("众病之王", "悉达多·穆克吉", "医学", "R73-49/0003", "4楼自然科学阅览区", 3, 1),
    # ---- 英语 ----
    ("新概念英语", "路易·亚历山大", "英语", "H31/0860", "3楼语言文学阅览区", 6, 4),
    ("词汇的力量", "诺曼·刘易斯", "英语", "H313/0442", "3楼语言文学阅览区", 4, 3),
    ("英语语法新思维", "张满胜", "英语", "H314/0288", "3楼语言文学阅览区", 4, 2),
    # ---- 日语 ----
    ("标日初级", "人民教育出版社", "日语", "H36/0154", "3楼语言文学阅览区", 5, 3),
    ("大家的日语", "株式会社スリーエーネットワーク", "日语", "H36/0155", "3楼语言文学阅览区", 4, 3),
    # ---- 社会学 ----
    ("社会心理学", "戴维·迈尔斯", "社会学", "B84/0760", "3楼社会科学阅览区", 4, 2),
    ("乌合之众", "古斯塔夫·勒庞", "社会学", "C912.64/0012", "3楼社会科学阅览区", 5, 4),
    ("乡土中国", "费孝通", "社会学", "C912.82/0005", "3楼社会科学阅览区", 4, 3),
    # ---- 心理 ----
    ("思考，快与慢", "丹尼尔·卡尼曼", "心理", "B842.5/0031", "3楼社会科学阅览区", 4, 2),
    ("被讨厌的勇气", "岸见一郎", "心理", "B821-49/0128", "3楼社会科学阅览区", 5, 4),
    ("非暴力沟通", "马歇尔·卢森堡", "心理", "C912.11/0197", "3楼社会科学阅览区", 4, 3),
    # ---- 小说 ----
    ("活着", "余华", "小说", "I247.5/0998", "2楼文学借阅区", 5, 3),
    ("百年孤独", "加西亚·马尔克斯", "小说", "I775.45/031", "2楼文学借阅区", 4, 2),
    ("平凡的世界", "路遥", "小说", "I247.5/0999", "2楼文学借阅区", 5, 4),
    # ---- 传记 ----
    ("乔布斯传", "沃尔特·艾萨克森", "传记", "K837.125.38/0021", "3楼历史地理阅览区", 4, 2),
    ("苏东坡传", "林语堂", "传记", "K825.6/0164", "3楼历史地理阅览区", 4, 3),
    ("富兰克林自传", "本杰明·富兰克林", "传记", "K837.127/0043", "3楼历史地理阅览区", 3, 2),
]

# 默认账号
DEFAULT_ADMIN = ("admin", "admin123", "admin", "系统管理员", "admin")
DEFAULT_READER_PASSWORD = "123456"


# ----------------------------------------------------------------------
# 建库建表
# ----------------------------------------------------------------------
def create_database() -> None:
    db_name = db_conf["database"]
    conn = get_server_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        logger.info(f"[init_db]数据库 {db_name} 已就绪")
    finally:
        conn.close()


def create_tables() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for ddl in (DDL_USERS, DDL_BOOKS, DDL_BORROW_RECORDS, DDL_BORROW_MONTHLY,
                        DDL_CONVERSATION_MEMORY, DDL_EVAL_TURN_RECORDS):
                cur.execute(ddl)
        logger.info("[init_db]6 张表已就绪")
    finally:
        conn.close()


# ----------------------------------------------------------------------
# 种子：用户
# ----------------------------------------------------------------------
def _read_records_csv() -> list[dict]:
    """读取 data/external/records.csv，返回逐行 dict。

    用 csv 模块而非 split(",")：原实现依赖「字段内不含逗号」这一脆弱前提。
    """
    csv_path = get_abs_path("data/external/records.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"借阅数据文件不存在：{csv_path}")

    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _parse_borrow_field(text: str, unit: str) -> int:
    """从「月借8册 | 阅读时长40h」里取出指定单位对应的数字。取不到时返回 0。

    必须先按 ``|`` 切分再定位含该单位的片段：直接抽取整串数字会把
    「4 册」和「21h」拼成 421。
    """
    for part in text.split("|"):
        if unit in part:
            digits = "".join(ch for ch in part if ch.isdigit())
            return int(digits) if digits else 0
    return 0


def seed_users(monthly_rows: list[dict]) -> int:
    """建默认管理员 + 10 位读者账号。

    读者显示名取自 CSV 偏好类别的前半段（如「文学爱好者 | 科幻/历史」→「文学爱好者」），
    这样管理后台的账号列表与借阅画像能对上。
    """
    # 读者显示名映射
    display_names: dict[str, str] = {}
    for row in monthly_rows:
        reader_id = row["用户ID"].strip()
        if reader_id not in display_names:
            display_names[reader_id] = row["偏好类别"].split("|")[0].strip()

    username, password, role, display_name, reader_id = DEFAULT_ADMIN
    count = execute(
        "INSERT IGNORE INTO users (username, password_hash, role, display_name, reader_id) "
        "VALUES (%s, %s, %s, %s, %s)",
        (username, hash_password(password), role, display_name, reader_id),
    )

    for rid in sorted(display_names):
        count += execute(
            "INSERT IGNORE INTO users (username, password_hash, role, display_name, reader_id) "
            "VALUES (%s, %s, 'reader', %s, %s)",
            (f"reader{rid}", hash_password(DEFAULT_READER_PASSWORD),
             display_names[rid], rid),
        )

    logger.info(f"[init_db]新增用户 {count} 个（已存在的跳过）")
    return count


# ----------------------------------------------------------------------
# 种子：图书
# ----------------------------------------------------------------------
def seed_books() -> int:
    count = 0
    for title, author, category, call_number, location, total, available in SEED_BOOKS:
        status = "在架可借" if available > 0 else "已借空"
        count += execute(
            "INSERT IGNORE INTO books (title, author, category, call_number, location, "
            "total_stock, available_stock, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (title, author, category, call_number, location, total, available, status),
        )
    logger.info(f"[init_db]新增图书 {count} 本（已存在的跳过）")
    return count


# ----------------------------------------------------------------------
# 种子：借阅
# ----------------------------------------------------------------------
def _iter_book_titles(current_books: str) -> list[str]:
    """从「《三体》《万历十五年》」拆出书名列表。"""
    titles = []
    for chunk in current_books.split("《"):
        name = chunk.split("》")[0].strip()
        if name:
            titles.append(name)
    return titles


def seed_borrow_monthly(monthly_rows: list[dict]) -> int:
    count = 0
    for row in monthly_rows:
        count += execute(
            "INSERT IGNORE INTO borrow_monthly "
            "(reader_id, month, preference, borrow_count, reading_hours, current_books, comparison) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                row["用户ID"].strip(),
                row["时间"].strip(),
                row["偏好类别"].strip(),
                _parse_borrow_field(row["月度借阅量"], "册"),
                _parse_borrow_field(row["月度借阅量"], "h"),
                row["在借图书"].strip(),
                row["对比分析"].strip(),
            ),
        )
    logger.info(f"[init_db]新增月度汇总 {count} 行（已存在的跳过）")
    return count


def seed_borrow_records(monthly_rows: list[dict]) -> int:
    """由月度汇总里的「在借图书」展开成借阅明细。"""
    existing = query_one("SELECT COUNT(*) AS c FROM borrow_records")
    if existing and existing["c"] > 0:
        logger.info(f"[init_db]借阅明细已存在 {existing['c']} 行，跳过")
        return 0

    # 2025-09 及以前视为已归还；10/11 月逾期未还；12 月在借
    returned_before = "2025-09"

    count = 0
    for row in monthly_rows:
        month = row["时间"].strip()
        reader_id = row["用户ID"].strip()
        year, mon = (int(x) for x in month.split("-"))
        borrow_date = date(year, mon, 5)
        due_date = borrow_date + timedelta(days=30)

        if month <= returned_before:
            status, return_date = "returned", borrow_date + timedelta(days=15)
        elif month < "2025-12":
            status, return_date = "overdue", None
        else:
            status, return_date = "borrowed", None

        for title in _iter_book_titles(row["在借图书"]):
            count += execute(
                "INSERT INTO borrow_records "
                "(reader_id, book_title, month, borrow_date, due_date, return_date, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (reader_id, title, month, borrow_date, due_date, return_date, status),
            )

    logger.info(f"[init_db]新增借阅明细 {count} 行")
    return count


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main() -> None:
    logger.info("[init_db]开始初始化图书馆数据库……")

    create_database()
    create_tables()

    monthly_rows = _read_records_csv()
    logger.info(f"[init_db]从 records.csv 读到 {len(monthly_rows)} 行月度记录")

    seed_users(monthly_rows)
    seed_books()
    seed_borrow_monthly(monthly_rows)
    seed_borrow_records(monthly_rows)

    # 汇总（质检表只统计不清空，行数取决于跑过多少轮对话）
    for table in ("users", "books", "borrow_records", "borrow_monthly",
                  "conversation_memory", "eval_turn_records"):
        row = query_one(f"SELECT COUNT(*) AS c FROM {table}")
        logger.info(f"[init_db]  {table}: {row['c']} 行")

    logger.info("[init_db]初始化完成。默认管理员 admin/admin123，读者 reader1001/123456")
    print("\n数据库初始化完成。")
    print("  管理员：admin / admin123")
    print("  读者　：reader1001 ~ reader1010 / 123456")


if __name__ == "__main__":
    main()
