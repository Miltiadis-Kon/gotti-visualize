"""
Tests for the new plot infrastructure:
  - PlotSpec / Layer dataclasses
  - ChartBuilder fluent API
  - PlotlyRenderer
  - Backward compatibility of chart_utils
"""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chart_df(n: int = 30) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "Date": dates,
        "open":  close * 0.99,
        "high":  close * 1.01,
        "low":   close * 0.98,
        "close": close,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    })
    return df


def make_levels_df() -> pd.DataFrame:
    """Create a synthetic levels DataFrame."""
    return pd.DataFrame({
        "level_price": [95.0, 100.0, 105.0, 110.0],
        "type":        ["support", "support", "resistance", "resistance"],
        "importance":  [3, 5, 4, 2],
        "touch_count": [3, 6, 4, 2],
    })


# ---------------------------------------------------------------------------
# PlotSpec / Layer tests
# ---------------------------------------------------------------------------

class TestPlotSpec:
    def test_candlestick_layer_creation(self):
        from plots.plot_spec import CandlestickLayer
        df = make_chart_df()
        layer = CandlestickLayer(data=df, ticker="TEST", timeframe="1D")
        assert layer.ticker == "TEST"
        assert layer.timeframe == "1D"
        assert layer.visible is True

    def test_key_levels_layer_defaults(self):
        from plots.plot_spec import KeyLevelsLayer
        df = make_levels_df()
        layer = KeyLevelsLayer(levels_df=df)
        assert layer.min_importance == 1
        assert layer.show_support is True
        assert layer.show_resistance is True
        assert layer.proximity_filter == 0.25

    def test_trade_markers_layer(self):
        from plots.plot_spec import TradeMarkersLayer
        layer = TradeMarkersLayer(trades=["mock_trade"])
        assert layer.show_tp_sl is True
        assert layer.show_zones is True

    def test_indicator_layer_defaults(self):
        from plots.plot_spec import IndicatorLayer
        layer = IndicatorLayer(name="SMA 50", data=None)
        assert layer.color == "#FFD700"
        assert layer.dash == "solid"
        assert layer.row == 1

    def test_plot_spec_candlestick_layer_accessor(self):
        from plots.plot_spec import PlotSpec, CandlestickLayer
        df = make_chart_df()
        cs = CandlestickLayer(data=df, ticker="NVDA")
        spec = PlotSpec(ticker="NVDA", layers=[cs])
        assert spec.candlestick_layer is cs

    def test_plot_spec_get_layers_of_type(self):
        from plots.plot_spec import PlotSpec, CandlestickLayer, KeyLevelsLayer
        df = make_chart_df()
        spec = PlotSpec(
            ticker="NVDA",
            layers=[
                CandlestickLayer(data=df, ticker="NVDA"),
                KeyLevelsLayer(levels_df=make_levels_df()),
            ],
        )
        cs_layers = spec.get_layers_of_type(CandlestickLayer)
        kl_layers = spec.get_layers_of_type(KeyLevelsLayer)
        assert len(cs_layers) == 1
        assert len(kl_layers) == 1

    def test_plot_spec_has_trades_false(self):
        from plots.plot_spec import PlotSpec, CandlestickLayer
        df = make_chart_df()
        spec = PlotSpec(ticker="T", layers=[CandlestickLayer(data=df, ticker="T")])
        assert spec.has_trades is False


# ---------------------------------------------------------------------------
# ChartBuilder tests
# ---------------------------------------------------------------------------

class TestChartBuilder:
    def test_basic_build(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        spec = ChartBuilder("NVDA", df, "1D").add_candlesticks().build()
        assert spec.ticker == "NVDA"
        assert spec.timeframe == "1D"
        assert len(spec.layers) == 1

    def test_build_without_candlesticks_raises(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        builder = ChartBuilder("NVDA", df)
        with pytest.raises(ValueError, match="at least one candlestick"):
            builder.build()

    def test_add_key_levels_empty_skipped(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .add_key_levels(pd.DataFrame())   # empty → should not add layer
            .build()
        )
        from plots.plot_spec import KeyLevelsLayer
        assert len(spec.get_layers_of_type(KeyLevelsLayer)) == 0

    def test_add_trades_empty_skipped(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .add_trades([])
            .build()
        )
        from plots.plot_spec import TradeMarkersLayer
        assert len(spec.get_layers_of_type(TradeMarkersLayer)) == 0

    def test_add_key_levels(self):
        from plots.chart_builder import ChartBuilder
        from plots.plot_spec import KeyLevelsLayer
        df = make_chart_df()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .add_key_levels(make_levels_df(), min_importance=3)
            .build()
        )
        kl = spec.get_layers_of_type(KeyLevelsLayer)
        assert len(kl) == 1
        assert kl[0].min_importance == 3

    def test_set_height_and_theme(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .set_height(900)
            .set_theme("tradingview_dark")
            .build()
        )
        assert spec.height == 900
        assert spec.theme == "tradingview_dark"

    def test_add_indicator(self):
        from plots.chart_builder import ChartBuilder
        from plots.plot_spec import IndicatorLayer
        df = make_chart_df()
        sma = df["close"].rolling(10).mean()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .add_indicator("SMA 10", sma, color="#FF0000")
            .build()
        )
        ind = spec.get_layers_of_type(IndicatorLayer)
        assert len(ind) == 1
        assert ind[0].name == "SMA 10"
        assert ind[0].color == "#FF0000"

    def test_from_plot_spec_round_trip(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        original_spec = (
            ChartBuilder("NVDA", df).add_candlesticks().set_height(800).build()
        )
        rebuilt_spec = ChartBuilder.from_plot_spec(original_spec).build()
        assert rebuilt_spec.height == 800
        assert rebuilt_spec.ticker == "NVDA"

    def test_set_extra(self):
        from plots.chart_builder import ChartBuilder
        df = make_chart_df()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .set_extra(rangeslider=False)
            .build()
        )
        assert spec.extra.get("rangeslider") is False


# ---------------------------------------------------------------------------
# PlotlyRenderer tests
# ---------------------------------------------------------------------------

class TestPlotlyRenderer:
    def test_render_returns_figure(self):
        import plotly.graph_objects as go
        from plots.chart_builder import ChartBuilder
        from plots.renderers import get_renderer
        df = make_chart_df()
        spec = ChartBuilder("NVDA", df).add_candlesticks().build()
        fig = get_renderer("plotly").render(spec)
        assert isinstance(fig, go.Figure)

    def test_render_has_candlestick_trace(self):
        import plotly.graph_objects as go
        from plots.chart_builder import ChartBuilder
        from plots.renderers import get_renderer
        df = make_chart_df()
        spec = ChartBuilder("NVDA", df).add_candlesticks().build()
        fig = get_renderer("plotly").render(spec)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Candlestick" in trace_types

    def test_render_key_levels_adds_scatter_traces(self):
        import plotly.graph_objects as go
        from plots.chart_builder import ChartBuilder
        from plots.renderers import get_renderer
        df = make_chart_df()
        levels = make_levels_df()
        spec = (
            ChartBuilder("NVDA", df)
            .add_candlesticks()
            .add_key_levels(levels, min_importance=1, proximity_filter=1.0)
            .build()
        )
        fig = get_renderer("plotly").render(spec)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # 4 levels → 4 scatter traces
        assert len(scatter_traces) == 4

    def test_render_invisible_layer_skipped(self):
        import plotly.graph_objects as go
        from plots.plot_spec import PlotSpec, CandlestickLayer, KeyLevelsLayer
        from plots.renderers import get_renderer
        df = make_chart_df()
        spec = PlotSpec(
            ticker="T",
            layers=[
                CandlestickLayer(data=df, ticker="T"),
                KeyLevelsLayer(
                    levels_df=make_levels_df(),
                    visible=False,           # ← should be skipped
                    proximity_filter=1.0,
                ),
            ],
        )
        fig = get_renderer("plotly").render(spec)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 0

    def test_layout_height(self):
        from plots.chart_builder import ChartBuilder
        from plots.renderers import get_renderer
        df = make_chart_df()
        spec = ChartBuilder("NVDA", df).add_candlesticks().set_height(500).build()
        fig = get_renderer("plotly").render(spec)
        assert fig.layout.height == 500

    def test_unknown_backend_raises(self):
        from plots.renderers import get_renderer
        with pytest.raises(ValueError, match="Unknown renderer backend"):
            get_renderer("nonexistent")


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_chart_utils_colors_accessible(self):
        from plots.chart_utils import COLORS
        assert "candle_up" in COLORS
        assert "candle_down" in COLORS

    def test_create_trade_chart_deprecated(self):
        from plots.chart_utils import create_trade_chart
        import plotly.graph_objects as go
        df = make_chart_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # create_trade_chart wraps ChartBuilder internally
            # It may raise ValueError if it still uses the old path —
            # we just check the DeprecationWarning is emitted
            try:
                fig = create_trade_chart("T", df)
            except Exception:
                pass
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

    def test_plots_init_exports_new_api(self):
        import plots
        assert hasattr(plots, "ChartBuilder")
        assert hasattr(plots, "get_renderer")
        assert hasattr(plots, "PlotSpec")
        assert hasattr(plots, "CandlestickLayer")

    def test_plots_init_exports_legacy_api(self):
        import plots
        assert hasattr(plots, "PLOT")
        assert hasattr(plots, "set_plot_enabled")


# ---------------------------------------------------------------------------
# PlottableStrategyMixin (unit test without full Lumibot stack)
# ---------------------------------------------------------------------------

class TestPlottableStrategyMixin:
    def test_render_plot_raises_without_get_plot_spec(self):
        from strategies.plot_mixin import PlottableStrategyMixin

        class BrokenStrategy(PlottableStrategyMixin):
            pass  # missing get_plot_spec

        s = BrokenStrategy()
        with pytest.raises(NotImplementedError):
            s.get_plot_spec()

    def test_render_plot_calls_renderer(self, tmp_path):
        from strategies.plot_mixin import PlottableStrategyMixin
        from plots.chart_builder import ChartBuilder

        df = make_chart_df()

        class MockStrategy(PlottableStrategyMixin):
            def get_plot_spec(self):
                return ChartBuilder("T", df).add_candlesticks().build()

        s = MockStrategy()
        # render without showing (headless)
        import plotly.graph_objects as go
        fig = s.render_plot(show=False)
        assert isinstance(fig, go.Figure)

    def test_save_plot_html(self, tmp_path):
        from strategies.plot_mixin import PlottableStrategyMixin
        from plots.chart_builder import ChartBuilder

        df = make_chart_df()
        output = tmp_path / "chart.html"

        class MockStrategy(PlottableStrategyMixin):
            def get_plot_spec(self):
                return ChartBuilder("T", df).add_candlesticks().build()

        s = MockStrategy()
        s.save_plot_html(str(output))
        assert output.exists()
        assert output.stat().st_size > 0
