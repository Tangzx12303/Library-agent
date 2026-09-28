"""指标计算。

一条贯穿全文件的原则：**分母为 0 时返回 None（不适用），绝不返回 0**。

「图书馆有哪些功能分区？」这类轮次根本没有图书推荐，推荐准确率是**不适用**，
不是「0% 准确」。这个区分不是洁癖——它决定了管理员的图表是否可信：
MySQL 的 ``AVG()`` 会自动跳过 NULL，所以只要落库时写 NULL，这些轮次就不会被
算进均值。若写 0，问答类型的分布一变（比如某天知识类问题变多），管理员看到的
推荐 F1 趋势线就会出现一次纯属虚构的塌陷。
"""


def ratio(numerator: int, denominator: int) -> float | None:
    """安全除法：分母为 0 返回 None。"""
    if not denominator:
        return None
    return numerator / denominator


def precision(predicted: set[str], truth: set[str]) -> float | None:
    """准确率 = 预测中命中的比例。

    分母是**预测集**大小；预测为空表示「本轮没有做出这类判断」，返回 None。
    """
    return ratio(len(predicted & truth), len(predicted))


def recall(predicted: set[str], truth: set[str]) -> float | None:
    """召回率 = 真值中被覆盖的比例。

    分母是**真值集**大小；真值为空表示「本轮没有可覆盖的对象」，返回 None。
    """
    return ratio(len(predicted & truth), len(truth))


def f1_score(p: float | None, r: float | None) -> float | None:
    """F1 = 调和平均。任一分量为 None 则整体不适用。"""
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def group_recall(actual: set[str], groups: tuple[tuple[str, ...], ...],
                 ) -> tuple[float | None, list[tuple[str, ...]]]:
    """按「OR 组」计算召回率，返回 ``(召回率, 未满足的组)``。

    每组是「命中其中任一工具即算满足」的候选集。这个结构是为了**不惩罚模型的
    合理解释性行为**：读者问「有哪些科幻类的书可以借」时，模型先调
    ``search_book("三体")`` 核实再调 ``search_books_by_category``，把两者放进
    同一个 OR 组，多调一个同类工具就不会被扣分。

    没有组（``other`` / ``failed`` 场景）时返回 None —— 不适用。
    """
    if not groups:
        return None, []

    unsatisfied = [group for group in groups if not (set(group) & actual)]
    return (len(groups) - len(unsatisfied)) / len(groups), unsatisfied


def compliance_rate(actual: set[str], forbidden: set[str]) -> float | None:
    """调用合规率 = 实际调用中「未被禁止」的比例。

    这里刻意用**禁用集**而不是期望集做基准。若拿期望集算准确率，模型多做一件
    合理的事就会被扣分，指标衡量的是「模型是否恰好和关键词表一样懒」而不是
    「答得好不好」。

    绝大多数场景禁用集为空，合规率恒为 1.0 —— 这正是重点：**只惩罚提示词明令
    禁止的调用**。本项目的实质规则只有一条：非报告场景禁用
    ``fill_context_for_report``（见 ``prompts/main_prompt.txt``）。
    """
    if not actual:
        return None
    return len(actual - forbidden) / len(actual)


def extra_call_rate(actual: set[str], required_flat: set[str]) -> float | None:
    """额外调用率 —— **信息性指标，不参与评分**。

    「该模型话痨 / 爱乱调工具」这件事值得看见，但不该被当成错误，所以只进详情
    和图表，不进任何通过率。
    """
    if not actual:
        return None
    return len(actual - required_flat) / len(actual)
