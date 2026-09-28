"""图表规格：配色按规则取用，且规格本身可序列化（能被 Streamlit 渲染）。"""
import json

import pandas as pd
import pytest

from quality import charts as ch


def _json(chart) -> str:
    return json.dumps(chart.to_dict(), ensure_ascii=False)


@pytest.fixture
def failure_frame():
    return pd.DataFrame({
        "check_name": ["无编造书目", "报告流程完整", "回答非空"],
        "determined": [10, 10, 10],
        "failed": [4, 1, 0],
        "failure_rate": [0.4, 0.1, 0.0],
    })


@pytest.fixture
def model_frame():
    return pd.DataFrame({
        "model_id": ["qwen3.8-max", "deepseek-v3.2"],
        "n": [20, 12],
        "pass_rate": [0.9, 0.75],
        "tool_recall": [0.95, 0.6],
        "rec_f1": [0.8, 0.7],
    })


@pytest.fixture
def trend_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-09-20", "2026-09-21", "2026-09-22"]),
        "n": [10, 4, 12],
        "pass_rate": [0.8, 0.5, 0.9],
    })


@pytest.fixture
def matrix_frame():
    return pd.DataFrame({
        "scenario_label": ["推荐报告", "推荐报告", "馆藏查询"],
        "check_name": ["报告流程完整", "回答非空", "回答非空"],
        "pass_rate": [0.5, 1.0, 0.9],
        "n": [4, 4, 10],
    })


# ----------------------------------------------------------------------
# 规格有效性
# ----------------------------------------------------------------------

def test_all_charts_produce_serializable_specs(failure_frame, model_frame,
                                               trend_frame, matrix_frame):
    """``to_dict()`` 能跑通 = Altair 规格合法，Streamlit 才画得出来。"""
    for chart in (ch.failure_rate_chart(failure_frame),
                  ch.model_comparison_chart(model_frame),
                  ch.pass_rate_trend_chart(trend_frame),
                  ch.sample_size_chart(trend_frame),
                  ch.scenario_check_heatmap(matrix_frame)):
        spec = chart.to_dict()
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/")


def _y_fields(chart) -> set[str]:
    """收集 spec 里所有 y 通道的字段名（含分层图与分组偏移通道）。"""
    spec = chart.to_dict()
    layers = spec.get("layer") or [spec]

    fields = set()
    for layer in layers:
        for channel in ("y", "yOffset"):
            encoding = (layer.get("encoding") or {}).get(channel)
            if isinstance(encoding, dict) and "field" in encoding:
                fields.add(encoding["field"])
    return fields


def test_trend_and_sample_size_are_two_charts_not_two_scales(trend_frame):
    """双 y 轴是最常见的图表错误：两个刻度的对齐是任意的，会凭空造出相关性。

    对策是把「通过率」和「样本量」拆成**两个各自只有一个 y 轴**的图，共享同一条
    x 轴。这条测试直接钉住这个决定 —— 两张图的 y 通道字段必须互不相干
    （tooltip 里带对方的值是允许的，那是补充信息，不是第二个刻度）。
    """
    rate_y = _y_fields(ch.pass_rate_trend_chart(trend_frame))
    size_y = _y_fields(ch.sample_size_chart(trend_frame))

    assert rate_y == {"pass_rate"}
    assert size_y == {"n"}


def test_model_comparison_has_one_shared_measure_axis(model_frame):
    """两个序列共用**同一个度量轴**（x = rate），分组靠 yOffset 通道。

    yOffset 是分组机制本身，不是第二个刻度 —— 两个序列量的都是「比率」，
    共用一把尺子，所以这张图不存在双轴问题。
    """
    spec = ch.model_comparison_chart(model_frame).to_dict()
    encoding = spec["encoding"]

    assert encoding["x"]["field"] == "rate"
    assert encoding["yOffset"]["field"] == "metric"
    assert "y" not in encoding or encoding["y"]["field"] == "model_id"
    # 只有一个颜色通道承载两个序列，且取自前两个固定槽位
    assert encoding["color"]["field"] == "metric"


# ----------------------------------------------------------------------
# 配色：按用途取色，不循环、不生成
# ----------------------------------------------------------------------

def test_single_series_uses_one_colour(failure_frame):
    """单序列必须单色。

    按失败率高低下色阶属于「给名义类别上色阶」，是反模式 —— 它把条长已经表达过
    的信息再用色相说一遍，白白烧掉唯一的自由通道。
    """
    spec = _json(ch.failure_rate_chart(failure_frame))

    assert ch.SERIES_1 in spec
    assert ch.SERIES_2 not in spec, "单序列图不该出现第二个分类色"


def test_grouped_chart_uses_exactly_two_slots(model_frame):
    spec = _json(ch.model_comparison_chart(model_frame))

    assert ch.SERIES_1 in spec and ch.SERIES_2 in spec


def test_trend_line_is_single_series(trend_frame):
    """通过率折线只有一条序列，因此不该有图例色块 —— 标题已经说明画的是什么。"""
    spec = _json(ch.pass_rate_trend_chart(trend_frame))

    assert ch.SERIES_1 in spec
    assert ch.SERIES_2 not in spec


def test_heatmap_uses_one_hue_sequential_ramp(matrix_frame):
    """顺序色阶必须单一色相由浅到深，绝不能是彩虹。"""
    spec = _json(ch.scenario_check_heatmap(matrix_frame))

    assert ch.BLUE_RAMP[0] in spec and ch.BLUE_RAMP[-1] in spec
    assert ch.SERIES_2 not in spec, "热力图不该混入第二个色相"
    assert "#e34948" not in spec and "#008300" not in spec


def test_palette_slots_are_taken_in_fixed_order():
    """分类色按固定槽位取用。本模块只用到前两槽，因此相邻对就是文档里已验证的那一对。"""
    assert ch.SERIES_1 == "#2a78d6"     # 槽 1 蓝
    assert ch.SERIES_2 == "#eb6834"     # 槽 2 橙
    assert len(ch.BLUE_RAMP) == 13


# ----------------------------------------------------------------------
# 图表底板
# ----------------------------------------------------------------------

def test_gridlines_are_solid_not_dashed(failure_frame):
    """虚线网格会读成「阈值」或「预测」，而它只是一条网格。"""
    spec = ch.failure_rate_chart(failure_frame).to_dict()

    assert spec["config"]["axis"]["gridDash"] == []


def test_axis_ink_is_recessive(failure_frame):
    """坐标轴与刻度用隐性墨色，网格是一档之隔的发丝线 —— 数据才是唯一该响的东西。"""
    axis = ch.failure_rate_chart(failure_frame).to_dict()["config"]["axis"]

    assert axis["labelColor"] == ch.INK_MUTED
    assert axis["gridColor"] == ch.GRIDLINE
    assert axis["gridWidth"] == 1
    assert axis["titleFont"] == ch.FONT


def test_chart_chrome_survives_the_padding_config(failure_frame):
    """回归：``.configure(padding=...)`` 会**重置整个 config**。

    它原本挂在配置链末尾，把前面设好的 axis / legend / title 配置全部丢掉，
    图表静默退回 Altair 默认外观（粗网格、带边框的坐标轴）—— 不报错，只是变丑。
    """
    config = ch.failure_rate_chart(failure_frame).to_dict()["config"]

    assert "padding" in config
    assert "axis" in config, "configure(padding) 把 axis 配置冲掉了"
    assert "legend" in config, "configure(padding) 把 legend 配置冲掉了"
    assert config["view"]["stroke"] is None, "图表不该有边框"


def test_bars_are_capped_thin(failure_frame, model_frame, trend_frame):
    """条厚有上限，余下的 band 宽度留作空白 —— 塞满会显得很吵。"""
    spec = ch.failure_rate_chart(failure_frame).to_dict()

    assert spec["layer"][0]["mark"]["size"] == ch.BAR_SIZE
    assert ch.BAR_SIZE <= 24
    assert ch.BAR_SIZE_GROUPED <= 24


def test_percent_axis_pins_ticks_within_100():
    """横条图留了 15% 余量给标签，刻度必须钉死，否则会画出 115% 这种不存在的量级。"""
    axis = ch._percent_axis()

    assert axis.values == [0, 0.25, 0.5, 0.75, 1.0]
    assert axis.format == "%"


# ----------------------------------------------------------------------
# 边界
# ----------------------------------------------------------------------

def test_single_model_still_renders(model_frame):
    """只有一个模型时，对比图仍应可用（不该退化成一根没有参照的条）。"""
    spec = ch.model_comparison_chart(model_frame.head(1)).to_dict()

    assert spec is not None


def test_single_day_trend_still_renders(trend_frame):
    """只有一天数据时，末点标注仍要给出 —— 折线退化成点也不能没有读数。"""
    spec = _json(ch.pass_rate_trend_chart(trend_frame.head(1)))

    assert "80%" in spec


def test_heatmap_with_single_cell(matrix_frame):
    spec = ch.scenario_check_heatmap(matrix_frame.head(1)).to_dict()

    assert spec is not None
