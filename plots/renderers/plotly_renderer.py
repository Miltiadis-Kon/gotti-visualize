"""
Plotly Chart Renderer — Concrete Implementation (Template Method Pattern)

Consolidates all Plotly visualization logic previously scattered across:
  - plots/chart_utils.py
  - plots/trade_dashboard.py  (lines 407-610)
  - plots/key_levels_plot.py  (plot_key_levels_plotly)
  - strategies/book_based/*/  (inline schedule_plot / plot methods)

This is the single source of truth for TradingView-dark-style Plotly charts.
"""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd
import plotly.graph_objects as go

from plots.renderers.base import BaseChartRenderer
from plots.plot_spec import (
    PlotSpec,
    CandlestickLayer,
    KeyLevelsLayer,
    TradeMarkersLayer,
    IndicatorLayer,
)

# ---------------------------------------------------------------------------
# TradingView-style colour palette (single definition for the entire project)
# ---------------------------------------------------------------------------

COLORS = {
    "background":       "#131722",
    "paper":            "#131722",
    "text":             "#d1d4dc",
    "text_secondary":   "#b2b5be",
    "grid":             "rgba(42, 46, 57, 0.6)",
    "candle_up":        "#26a69a",
    "candle_down":      "#ef5350",
    "support":          "rgba(38, 166, 154, 0.9)",
    "resistance":       "rgba(239, 83, 80, 0.9)",
    "tp_line":          "rgba(38, 166, 154, 0.8)",
    "sl_line":          "rgba(239, 83, 80, 0.8)",
    "trade_win":        "rgba(38, 166, 154, 0.15)",
    "trade_loss":       "rgba(239, 83, 80, 0.15)",
    "entry_marker":     "#26a69a",
    "exit_marker":      "#ef5350",
    "fib_382":          "rgba(255, 215, 0, 0.7)",
    "fib_50":           "rgba(255, 165, 0, 0.7)",
    "fib_618":          "rgba(255, 99, 71, 0.7)",
}


class PlotlyRenderer(BaseChartRenderer):
    """Concrete Plotly renderer.

    Produces ``plotly.graph_objects.Figure`` objects styled in
    TradingView dark mode.
    """

    # ------------------------------------------------------------------
    # BaseChartRenderer hooks
    # ------------------------------------------------------------------

    def _create_figure(self, spec: PlotSpec) -> go.Figure:
        return go.Figure()

    # ---- Candlestick -------------------------------------------------

    def _render_candlestick(
        self, fig: go.Figure, layer: CandlestickLayer, spec: PlotSpec
    ) -> None:
        df = self._prepare_df(layer.data)
        fig.add_trace(
            go.Candlestick(
                x=df["date_plot"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=layer.ticker or spec.ticker,
                increasing_line_color=COLORS["candle_up"],
                decreasing_line_color=COLORS["candle_down"],
                increasing_fillcolor=COLORS["candle_up"],
                decreasing_fillcolor=COLORS["candle_down"],
            )
        )

    # ---- Key Levels --------------------------------------------------

    def _render_key_levels(
        self, fig: go.Figure, layer: KeyLevelsLayer, spec: PlotSpec
    ) -> None:
        df = layer.levels_df
        if df is None or df.empty:
            return

        # Determine x-axis bounds from the candlestick layer
        x_start, x_end = self._get_x_bounds(spec)

        # Current price for proximity filter
        candle = spec.candlestick_layer
        if candle is not None and candle.data is not None:
            prepared = self._prepare_df(candle.data)
            current_price = prepared["close"].iloc[-1]
        else:
            current_price = df["level_price"].mean()

        prox = layer.proximity_filter
        min_price = current_price * (1 - prox)
        max_price = current_price * (1 + prox)

        # Filter
        mask = (
            (df["importance"] >= layer.min_importance)
            & (df["level_price"] >= min_price)
            & (df["level_price"] <= max_price)
        )
        filtered = df[mask]

        for _, row in filtered.iterrows():
            level_type = row["type"]
            if level_type == "support" and not layer.show_support:
                continue
            if level_type == "resistance" and not layer.show_resistance:
                continue

            color = COLORS["support"] if level_type == "support" else COLORS["resistance"]
            dash  = "solid" if row["importance"] >= 4 else "dash"
            width = 2 if row["importance"] >= 4 else 1

            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[row["level_price"], row["level_price"]],
                    mode="lines+text",
                    line=dict(color=color, width=width, dash=dash),
                    text=["", f"${row['level_price']:.2f}"],
                    textposition="middle right",
                    textfont=dict(color=color, size=10),
                    name=f"${row['level_price']:.2f}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>${row['level_price']:.2f}</b><br>"
                        f"{row['type'].title()}<br>"
                        f"Touches: {row['touch_count']}<br>"
                        f"Importance: {row['importance']}"
                        "<extra></extra>"
                    ),
                )
            )

    # ---- Trade Markers -----------------------------------------------

    def _render_trades(
        self, fig: go.Figure, layer: TradeMarkersLayer, spec: PlotSpec
    ) -> None:
        if not layer.trades:
            return

        _, x_end = self._get_x_bounds(spec)

        for trade in layer.trades:
            entry_date = trade.date_executed
            exit_date  = trade.date_completed if trade.date_completed else x_end

            # ── Zone shading ──────────────────────────────────────────
            if layer.show_zones and trade.date_completed:
                zone_color = COLORS["trade_win"] if trade.is_winner else COLORS["trade_loss"]
                fig.add_vrect(
                    x0=entry_date,
                    x1=exit_date,
                    fillcolor=zone_color,
                    layer="below",
                    line_width=0,
                )

            # ── TP / SL lines ─────────────────────────────────────────
            if layer.show_tp_sl:
                fig.add_trace(go.Scatter(
                    x=[entry_date, exit_date],
                    y=[trade.take_profit, trade.take_profit],
                    mode="lines",
                    line=dict(color=COLORS["tp_line"], width=1, dash="dash"),
                    showlegend=False,
                    hovertemplate=f"<b>Take Profit</b><br>${trade.take_profit:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=[entry_date, exit_date],
                    y=[trade.stop_loss, trade.stop_loss],
                    mode="lines",
                    line=dict(color=COLORS["sl_line"], width=1, dash="dash"),
                    showlegend=False,
                    hovertemplate=f"<b>Stop Loss</b><br>${trade.stop_loss:.2f}<extra></extra>",
                ))

            # ── Entry marker ──────────────────────────────────────────
            entry_label = "BUY" if trade.trade_type == "BUY" else "SELL"
            fig.add_trace(go.Scatter(
                x=[entry_date],
                y=[trade.entry_price],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up" if trade.trade_type == "BUY" else "triangle-down",
                    size=14,
                    color=COLORS["entry_marker"],
                    line=dict(width=1, color="white"),
                ),
                text=[f"{entry_label} ${trade.entry_price:.2f}"],
                textposition="top center",
                textfont=dict(color=COLORS["entry_marker"], size=10),
                showlegend=False,
                hovertemplate=(
                    f"<b>ENTRY ({entry_label})</b><br>"
                    f"Price: ${trade.entry_price:.2f}<br>"
                    f"Qty: {trade.quantity}<br>"
                    f"TP: ${trade.take_profit:.2f}<br>"
                    f"SL: ${trade.stop_loss:.2f}"
                    "<extra></extra>"
                ),
            ))

            # ── Exit marker ───────────────────────────────────────────
            if trade.date_completed and trade.exit_price:
                exit_color = COLORS["entry_marker"] if trade.is_winner else COLORS["exit_marker"]
                pnl_text   = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"

                fig.add_trace(go.Scatter(
                    x=[trade.date_completed],
                    y=[trade.exit_price],
                    mode="markers+text",
                    marker=dict(
                        symbol="triangle-down" if trade.trade_type == "BUY" else "triangle-up",
                        size=14,
                        color=exit_color,
                        line=dict(width=1, color="white"),
                    ),
                    text=[f"{trade.exit_reason} {pnl_text}"],
                    textposition="bottom center",
                    textfont=dict(color=exit_color, size=10),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>EXIT ({trade.exit_reason})</b><br>"
                        f"Price: ${trade.exit_price:.2f}<br>"
                        f"PnL: {pnl_text}<br>"
                        f"Duration: {trade.duration:.1f}h"
                        "<extra></extra>"
                    ),
                ))

                # ── Entry-to-exit connection line ─────────────────────
                if layer.show_connection_line:
                    fig.add_trace(go.Scatter(
                        x=[entry_date, trade.date_completed],
                        y=[trade.entry_price, trade.exit_price],
                        mode="lines",
                        line=dict(
                            color=COLORS["entry_marker"] if trade.is_winner else COLORS["exit_marker"],
                            width=1,
                            dash="dot",
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

    # ---- Indicator ---------------------------------------------------

    def _render_indicator(
        self, fig: go.Figure, layer: IndicatorLayer, spec: PlotSpec
    ) -> None:
        if layer.data is None:
            return

        data = layer.data
        if isinstance(data, pd.DataFrame):
            # Use the first column if a DataFrame is passed
            data = data.iloc[:, 0]

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode="lines",
            name=layer.name,
            line=dict(color=layer.color, width=layer.width, dash=layer.dash),
            showlegend=True,
        ))

    # ---- Layout ------------------------------------------------------

    def _apply_layout(self, fig: go.Figure, spec: PlotSpec) -> None:
        # Build title
        if spec.title:
            title_text = spec.title
        else:
            candle = spec.candlestick_layer
            if candle is not None and candle.data is not None:
                df = self._prepare_df(candle.data)
                current_price = df["close"].iloc[-1]
                title_text = f"{spec.ticker} • {spec.timeframe} • ${current_price:.2f}"
            else:
                title_text = spec.ticker

        # Tick format depends on timeframe
        if spec.timeframe.lower() in ("15m", "5m", "1h"):
            tick_format = "%Y-%m-%d\n%H:%M"
        else:
            tick_format = "%Y-%m-%d"

        # Y-axis range from candlestick data
        y_range = None
        candle = spec.candlestick_layer
        if candle is not None and candle.data is not None:
            df = self._prepare_df(candle.data)
            y_range = [df["low"].min() * 0.98, df["high"].max() * 1.02]

        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=20, color=COLORS["text"], family="Arial"),
                x=0.01,
                xanchor="left",
            ),
            xaxis_title="",
            yaxis_title="",
            template="plotly_dark",
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["paper"],
            font=dict(family="Arial", size=11, color=COLORS["text_secondary"]),
            xaxis=dict(
                gridcolor=COLORS["grid"],
                showgrid=True,
                zeroline=False,
                rangeslider=dict(visible=False),
                showticklabels=True,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=False,
                tickformat=tick_format,
            ),
            yaxis=dict(
                side="right",
                gridcolor=COLORS["grid"],
                showgrid=True,
                zeroline=False,
                tickprefix="$",
                tickformat=".2f",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                **({"range": y_range} if y_range else {}),
            ),
            legend=dict(visible=False),
            height=spec.height,
            margin=dict(l=10, r=60, t=50, b=20),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#1e222d",
                font_size=12,
                font_family="Arial",
                bordercolor="#2a2e39",
            ),
            **spec.extra,
        )

    # ------------------------------------------------------------------
    # show() override
    # ------------------------------------------------------------------

    def show(self, fig: go.Figure) -> None:
        """Display the figure in the browser."""
        fig.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a ``date_plot`` column and lowercase OHLCV."""
        df = df.copy()

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Build date_plot from common column / index patterns
        if "date_plot" not in df.columns:
            for col in ("date", "datetime", "timestamp"):
                if col in df.columns:
                    df["date_plot"] = pd.to_datetime(df[col])
                    break
            else:
                df["date_plot"] = pd.to_datetime(df.index)

        return df

    @staticmethod
    def _get_x_bounds(spec: PlotSpec):
        """Return the x-axis start/end from the first CandlestickLayer."""
        candle = spec.candlestick_layer
        if candle is None or candle.data is None:
            return None, None

        df = candle.data.copy()
        df.columns = [c.lower() for c in df.columns]

        for col in ("date", "datetime", "timestamp", "date_plot"):
            if col in df.columns:
                dates = pd.to_datetime(df[col])
                return dates.iloc[0], dates.iloc[-1]

        dates = pd.to_datetime(df.index)
        return dates[0], dates[-1]
