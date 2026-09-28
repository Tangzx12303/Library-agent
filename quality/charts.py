"""管理员看板的图表定义 —— 纯 Altair 规格，不含 Streamlit，可离线断言。

配色沿用 dataviz 规范的中性色板（app 自身的主色 ``#3a6ea5`` 保持不动，
不重绘应用；色板只管**图表标记**）：

- 分类色按固定槽位取用，**不循环、不生成第 9 色**。本模块最多用到前两槽；
- 单序列图一律用槽 1，**不给同一序列内的不同类别上色阶** —— 那会把条长这个
  已经表达过的信息再用色相编码一遍，白白烧掉唯一的自由通道；
- 顺序色阶（热力图）用单一蓝色由浅到深，绝不用彩虹；
- 网格线与坐标轴是**实线发丝线**（虚线会读成"阈值"或"预测"，是噪声）。

Altair 已随 Streamlit 安装，因此本模块**零新增依赖**。
"""
import altair as alt
import pandas as pd

# ---- 分类色（固定槽位，按需取用，不循环）----
SERIES_1 = "#2a78d6"          # 蓝
SERIES_2 = "#eb6834"          # 橙

# ---- 单色顺序阶（浅 → 深），用于热力图这类连续量级 ----
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]

# ---- 图表底板与墨色 ----
SURFACE = "#ffffff"           # 与 .streamlit/config.toml 的 secondaryBackgroundColor 一致
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = 'system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif'

# 条厚上限（规范要求 ≤ 24px，留白由 band 的余量给出）
BAR_SIZE = 18
BAR_SIZE_GROUPED = 12

# 横条标签需要留出的余量：x 轴留 15% 余量，否则贴近 100% 的标签会被裁掉
_LABEL_HEADROOM_DOMAIN = [0, 1.15]
_LABEL_TICKS = [0, 0.25, 0.5, 0.75, 1.0]


def _base(chart: alt.Chart) -> alt.Chart:
    """统一的底板：发丝实线网格、隐性坐标轴、无边框。

    ⚠️ **``.configure(padding=...)`` 必须放在链首。** 它会把整个 ``config``
    重置掉，之后再用就不只是加 padding —— 前面设好的 ``configure_axis`` /
    ``configure_legend`` / ``configure_title`` 会被一起丢弃，图表静默退回
    Altair 默认外观（粗网格、深色坐标轴、带边框）。这个坑由
    ``tests/test_charts.py::test_gridlines_are_solid_not_dashed`` 钉住。
    """
    return (chart
            .configure(padding={"left": 4, "right": 16, "top": 8, "bottom": 4})
            .configure_view(stroke=None, fill=SURFACE)
            .configure_axis(
                grid=True, gridColor=GRIDLINE, gridWidth=1, gridDash=[],
                domainColor=BASELINE, domainWidth=1,
                tickColor=BASELINE, tickSize=4,
                labelColor=INK_MUTED, labelFont=FONT, labelFontSize=11,
                titleColor=INK_SECONDARY, titleFont=FONT, titleFontSize=11,
                labelFontWeight="normal")
            .configure_legend(
                labelColor=INK_SECONDARY, labelFont=FONT, labelFontSize=11,
                titleColor=INK_SECONDARY, titleFont=FONT, titleFontSize=11,
                symbolType="square", symbolSize=90, orient="top", direction="horizontal")
            .configure_title(font=FONT, fontSize=13, color=INK_PRIMARY,
                             anchor="start", fontWeight="bold"))


def _percent_axis(scale_domain=None) -> alt.Axis:
    """百分比坐标轴。

    ``values`` 显式钉死刻度：留了 15% 余量后，默认刻度会画出 115% 这种不存在的
    x 值。钉死刻度既保住了标签空间，又不会造出不存在的量级。
    """
    return alt.Axis(format="%", title=None, values=_LABEL_TICKS)


# ----------------------------------------------------------------------
# 1. 检查项失败分布（全页信息量最高的一张）
# ----------------------------------------------------------------------
def failure_rate_chart(data: pd.DataFrame) -> alt.Chart:
    """横向条形图：y = 检查项（按失败率降序），x = 失败率。

    单序列 → **单色**。按失败率深浅上色属于「给名义类别上色阶」，是反模式：
    它把条长已经表达过的信息再用色相说一遍。

    条末直接标注百分比 —— 读者要的是"哪一项最容易失败"，不必去比对坐标轴。
    """
    frame = data.copy()
    frame["label"] = (frame["failure_rate"] * 100).round(1).astype(str) + "%"

    bars = (alt.Chart(frame)
            .mark_bar(size=BAR_SIZE, cornerRadiusEnd=4, color=SERIES_1)
            .encode(
                x=alt.X("failure_rate:Q", axis=_percent_axis(),
                        scale=alt.Scale(domain=_LABEL_HEADROOM_DOMAIN)),
                y=alt.Y("check_name:N", title=None,
                        sort=alt.SortField("failure_rate", order="descending")),
                tooltip=[
                    alt.Tooltip("check_name:N", title="检查项"),
                    alt.Tooltip("failure_rate:Q", title="失败率", format=".1%"),
                    alt.Tooltip("failed:Q", title="失败轮次"),
                    alt.Tooltip("determined:Q", title="判定轮次"),
                ]))

    labels = (alt.Chart(frame)
              .mark_text(align="left", dx=6, font=FONT, fontSize=11,
                         color=INK_SECONDARY)
              .encode(x="failure_rate:Q", y=alt.Y("check_name:N", sort=alt.SortField(
                  "failure_rate", order="descending")), text="label:N"))

    return _base(alt.layer(bars, labels).properties(height=max(160, 34 * len(frame))))


# ----------------------------------------------------------------------
# 2. 按模型对比（双序列 → 必须带图例）
# ----------------------------------------------------------------------
def model_comparison_chart(data: pd.DataFrame) -> alt.Chart:
    """分组横向条形图：工具召回率 vs 全项通过率。

    这是唯一花掉双序列色额度的图 —— 两个序列都是比率，共用一个 x 轴，
    因此**不存在双 y 轴问题**。
    """
    melted = data.melt(
        id_vars=["model_id", "n"],
        value_vars=["tool_recall", "pass_rate"],
        var_name="metric", value_name="rate").dropna(subset=["rate"])
    melted["metric"] = melted["metric"].map(
        {"tool_recall": "工具召回率", "pass_rate": "全项通过率"})

    order = list(data["model_id"])

    bars = (alt.Chart(melted)
            .mark_bar(size=BAR_SIZE_GROUPED, cornerRadiusEnd=4)
            .encode(
                x=alt.X("rate:Q", axis=_percent_axis(),
                        scale=alt.Scale(domain=_LABEL_HEADROOM_DOMAIN)),
                y=alt.Y("model_id:N", title=None, sort=order),
                yOffset=alt.YOffset("metric:N", title=None),
                color=alt.Color("metric:N", title=None,
                                scale=alt.Scale(domain=["工具召回率", "全项通过率"],
                                                range=[SERIES_1, SERIES_2])),
                tooltip=[
                    alt.Tooltip("model_id:N", title="模型"),
                    alt.Tooltip("metric:N", title="指标"),
                    alt.Tooltip("rate:Q", title="值", format=".1%"),
                    alt.Tooltip("n:Q", title="样本轮次"),
                ]))

    return _base(bars.properties(height=max(140, 54 * len(order) + 40)))


# ----------------------------------------------------------------------
# 3. 趋势：通过率 + 样本量（两张图共享 x 轴，绝不合并成双 y 轴）
# ----------------------------------------------------------------------
def pass_rate_trend_chart(data: pd.DataFrame) -> alt.Chart:
    """通过率折线。

    单序列 → **不画图例**：只有一种颜色，标题已经说明画的是什么，一个色块的图例
    只是把标题重说一遍还占地方。

    只在**末点**直接标注 —— 每个点都标数字会糊成一片，反而没人读。
    """
    frame = data.dropna(subset=["pass_rate"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["label"] = (frame["pass_rate"] * 100).round(0).astype("Int64").astype(str) + "%"

    line = (alt.Chart(frame)
            .mark_line(strokeWidth=2, color=SERIES_1,
                       interpolate="monotone", point=False)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d",
                                                            labelColor=INK_MUTED,
                                                            titleColor=INK_SECONDARY)),
                y=alt.Y("pass_rate:Q", title=None, axis=_percent_axis(),
                        scale=alt.Scale(domain=[0, 1.15])),
                tooltip=[alt.Tooltip("date:T", title="日期"),
                         alt.Tooltip("pass_rate:Q", title="通过率", format=".1%"),
                         alt.Tooltip("n:Q", title="轮次")]))

    # 末点标记 + 标注。≥8px 的点，带 2px 底板描边使其压在线上也读得清。
    last = frame.tail(1)
    end_dot = (alt.Chart(last)
               .mark_point(size=70, filled=True, color=SERIES_1,
                           stroke=SURFACE, strokeWidth=2)
               .encode(x="date:T", y="pass_rate:Q"))
    end_label = (alt.Chart(last)
                 .mark_text(align="left", dx=8, dy=-6, font=FONT,
                            fontSize=11, color=INK_SECONDARY)
                 .encode(x="date:T", y="pass_rate:Q", text="label:N"))

    return _base(alt.layer(line, end_dot, end_label)
                 .properties(height=170))


def sample_size_chart(data: pd.DataFrame) -> alt.Chart:
    """每日轮次（细条形图），与上面的通过率**共享同一条 x 轴**。

    为什么坚持拆成两张而不是画成双 y 轴：通过率离了样本量根本读不懂 —— 某天
    1 轮全过是 100%，40 轮全过也是 100%，只有下面这张 n 图能把两者区分开，
    而且不必凭空捏造两个刻度之间的对应关系。
    """
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    bars = (alt.Chart(frame)
            .mark_bar(size=14, cornerRadiusEnd=3, color=SERIES_1)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d",
                                                            labelColor=INK_MUTED,
                                                            titleColor=INK_SECONDARY)),
                y=alt.Y("n:Q", title=None, axis=alt.Axis(tickMinStep=1,
                                                         labelColor=INK_MUTED)),
                tooltip=[alt.Tooltip("date:T", title="日期"),
                         alt.Tooltip("n:Q", title="轮次")]))

    return _base(bars.properties(height=110, title="每日样本量"))


# ----------------------------------------------------------------------
# 4. 场景 × 检查项 热力图
# ----------------------------------------------------------------------
def scenario_check_heatmap(data: pd.DataFrame) -> alt.Chart:
    """通过率矩阵，单一蓝色顺序阶。

    这是**发现规则错误**的地方：某个场景整列都在同一项上失败时，先怀疑
    「该场景的必需工具组定义错了」，而不是模型不行。

    单元格内不写字：格子通常放不下「100% + n」还不被裁。值交给 tooltip 和
    下方的表格双胞胎 —— 规范里说了，标签放不下就别硬塞，宁可移出去。
    """
    checks = list(data.groupby("check_name")["n"].sum()
                  .sort_values(ascending=False).index)
    scenarios = list(data.groupby("scenario_label")["n"].sum()
                     .sort_values(ascending=False).index)

    cells = (alt.Chart(data)
             .mark_rect(stroke=SURFACE, strokeWidth=2)     # 2px 底板缝隙分隔色块
             .encode(
                 x=alt.X("check_name:N", title=None, sort=checks,
                         axis=alt.Axis(labelAngle=-35, labelLimit=180)),
                 y=alt.Y("scenario_label:N", title=None, sort=scenarios),
                 color=alt.Color("pass_rate:Q", title="通过率",
                                 scale=alt.Scale(range=BLUE_RAMP, domain=[0, 1]),
                                 legend=alt.Legend(format=".0%")),
                 tooltip=[alt.Tooltip("scenario_label:N", title="场景"),
                          alt.Tooltip("check_name:N", title="检查项"),
                          alt.Tooltip("pass_rate:Q", title="通过率", format=".1%"),
                          alt.Tooltip("n:Q", title="轮次")]))

    return _base(cells.properties(height=max(120, 40 * len(scenarios) + 80)))
