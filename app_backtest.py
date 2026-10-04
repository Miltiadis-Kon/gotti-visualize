"""
Gotti Backtest Runner
=====================
Streamlit web UI to run any strategy on any tickers over any date range.
Generates interactive Plotly HTML charts and a unified dashboard — identical
to the ones produced by the liquidity sweep simulation scripts.

Launch:
    .\\venv\\Scripts\\streamlit.exe run app_backtest.py
"""

import sys
import os
import io
import time
import queue
import threading
import subprocess
import tempfile
import json
from datetime import datetime, timedelta, date
from pathlib import Path

# ── Encoding fix for Windows ──────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Lumibot progress bar silence ──────────────────────────────────────────────
import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *a, **k: None
import lumibot.data_sources.data_source_backtesting
lumibot.data_sources.data_source_backtesting.print_progress_bar = lambda *a, **k: None

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lumibot.entities import Asset
from lumibot.backtesting import PandasDataBacktesting, YahooDataBacktesting

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from plots.mysql_candle_feed import candle_feed
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.swing.hcr_breakout_long import HCRBreakoutLongSwing
from strategies.swing.hcr_breakout_short import HCRBreakoutShortSwing
from strategies.swing.fib_retrace_swing import FibRetraceSwing
from strategies.swing.gap_setup_long import GapSetupLongSwing
from strategies.swing.gap_setup_short import GapSetupShortSwing
from strategies.swing.retrace_long import RetraceLongSwing
from strategies.swing.retrace_short import RetraceShortSwing
from strategies.swing.rsi_setup import RSISetupSwing
from strategies.multi_tf_strategy import MultiTimeframeKeyLevelsStrategy

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
CHARTS_DIR = REPO_ROOT / "logs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_REGISTRY = {
    "Institutional Liquidity Sweep (5-Min Intraday)": {
        "cls": LiquiditySweepStrategy,
        "timeframe": "5 min",
        "datasource": "mysql",
        "rth_only": True,
        "description": "Sweep-and-Reclaim institutional algorithm on 5-minute RTH candles. "
                       "Targets Previous Day High/Low, Opening Range, and gap-adjusted levels. "
                       "Uses ATR-buffered stop loss below/above the sweep wick.",
    },
    "HCR Breakout Long (Swing)": {
        "cls": HCRBreakoutLongSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "15-day Horizontal Consolidation Region breakout to the upside. "
                       "TP = Entry + 2×ATR, SL = Entry − 1.5×ATR.",
    },
    "HCR Breakout Short (Swing)": {
        "cls": HCRBreakoutShortSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "15-day HCR breakdown to the downside with mirror risk rules.",
    },
    "Fibonacci Retrace Swing": {
        "cls": FibRetraceSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "Buys retracements to the 61.8% Fibonacci level in an established uptrend.",
    },
    "Gap Setup Long (Swing)": {
        "cls": GapSetupLongSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "Enters on bullish gap-up opens above prior resistance with volume confirmation.",
    },
    "Gap Setup Short (Swing)": {
        "cls": GapSetupShortSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "Enters on bearish gap-down opens below prior support.",
    },
    "Retrace Long (Swing)": {
        "cls": RetraceLongSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "Buys pullbacks to key moving averages in confirmed uptrends.",
    },
    "Retrace Short (Swing)": {
        "cls": RetraceShortSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "Shorts bounces to key moving averages in confirmed downtrends.",
    },
    "RSI Setup (Swing)": {
        "cls": RSISetupSwing,
        "timeframe": "1 day",
        "datasource": "yahoo",
        "rth_only": False,
        "description": "Mean-reversion entries based on RSI extremes with trend-direction filter.",
    },
    "Multi-Timeframe Key Levels (5-Min)": {
        "cls": MultiTimeframeKeyLevelsStrategy,
        "timeframe": "5 min",
        "datasource": "mysql",
        "rth_only": True,
        "description": "Combines multi-resolution S/R key levels (1D, 4H, 15m) with Fibonacci retracements "
                       "on 5-minute RTH candles. Fast in-memory multi-timeframe resampling, ATR-buffered "
                       "bracket orders, and R:R >= 1.5 trade execution.",
    },
}

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gotti Backtest Runner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Institutional CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #131722; color: #d1d4dc; }
    [data-testid="stSidebar"] { background: #1e222d; border-right: 1px solid #2a2e39; }
    [data-testid="stSidebar"] * { color: #d1d4dc; }
    h1, h2, h3 { color: #ffffff; }
    .metric-card {
        background: #1e222d; border: 1px solid #2a2e39; border-radius: 8px;
        padding: 16px 20px; text-align: center;
    }
    .metric-card .label { font-size: 0.78em; color: #787b86; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.6em; font-weight: 700; margin-top: 4px; }
    .pos { color: #26a69a; }
    .neg { color: #ef5350; }
    .neu { color: #f39c12; }
    .stButton > button {
        background: #1e88e5; color: white; font-weight: 700;
        border: none; border-radius: 6px; padding: 10px 28px;
        font-size: 1em; width: 100%; cursor: pointer;
    }
    .stButton > button:hover { background: #1565c0; }
    .log-box {
        background: #0d1117; border: 1px solid #2a2e39; border-radius: 6px;
        font-family: "Consolas", monospace; font-size: 0.82em; color: #8b949e;
        padding: 14px; max-height: 320px; overflow-y: auto; white-space: pre-wrap;
    }
    div[data-testid="stSelectbox"] > div { background: #1e222d; border: 1px solid #2a2e39; }
    div[data-testid="stMultiSelect"] > div { background: #1e222d; }
    .result-table { width: 100%; border-collapse: collapse; }
    .result-table th {
        background: #2a2e39; color: #9598a1; font-size: 0.82em;
        text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 14px; text-align: left;
    }
    .result-table td { padding: 10px 14px; border-bottom: 1px solid #2a2e39; }
    .result-table tr:hover td { background: #252a36; }
    hr { border-color: #2a2e39; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Backtest Configuration")
    st.markdown("---")

    # Strategy Selector
    strategy_name = st.selectbox(
        "Strategy",
        list(STRATEGY_REGISTRY.keys()),
        help="Select the strategy algorithm to backtest."
    )
    cfg = STRATEGY_REGISTRY[strategy_name]

    st.markdown(
        f"<div style='color:#9598a1;font-size:0.82em;background:#131722;"
        f"border:1px solid #2a2e39;border-radius:6px;padding:10px;margin-top:4px;'>"
        f"{cfg['description']}</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Ticker Input
    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value="NVDA, PLTR, MARA, GLD",
        help="e.g. NVDA, AAPL, TSLA, GLD"
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    st.markdown("---")

    # Date Range
    today = date.today()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=90)

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start Date", value=default_start, max_value=default_end)
    with col_e:
        end_date = st.date_input("End Date", value=default_end, min_value=start_date, max_value=today)

    st.markdown("---")

    # Starting Budget
    budget = st.number_input(
        "Starting Budget ($)",
        min_value=1000,
        max_value=10_000_000,
        value=10_000,
        step=1000,
        help="Portfolio starting cash for each ticker run."
    )

    # Data lookback buffer for indicators
    lookback_days = st.slider(
        "Data Lookback Buffer (days before start)",
        min_value=7,
        max_value=60,
        value=20,
        help="Extra days of data fetched before the backtest start to warm up indicators (ATR, VWAP)."
    )

    st.markdown("---")
    run_button = st.button("🚀 Run Backtest")


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("# 📈 Gotti Backtest Runner")
st.markdown(
    f"**Strategy:** `{strategy_name}` &nbsp;|&nbsp; "
    f"**Timeframe:** `{cfg['timeframe']}` &nbsp;|&nbsp; "
    f"**Data Source:** `{'MySQL (Local)' if cfg['datasource'] == 'mysql' else 'Yahoo Finance'}`"
)
st.markdown("---")

# ── Helper: Generate Plotly Chart ─────────────────────────────────────────────
def generate_plotly_chart(ticker, df, trades, metrics, output_path):
    """Build interactive institutional OHLCV chart with trade overlays."""
    bt_start_str = str(start_date)
    bt_end_str = str(end_date + timedelta(days=1))

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        plot_df = df[(df.index >= bt_start_str) & (df.index <= bt_end_str)].copy()
    else:
        try:
            plot_df = df[(df.index >= bt_start_str) & (df.index <= bt_end_str)].copy()
        except Exception:
            plot_df = df.copy()

    if plot_df.empty:
        plot_df = df.copy()

    ret_sign = f"+{metrics['return_pct']:.2f}" if metrics["return_pct"] >= 0 else f"{metrics['return_pct']:.2f}"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(
            f"<b>{ticker}</b> {cfg['timeframe']} · {strategy_name} | "
            f"Return: {ret_sign}% | Win Rate: {metrics['win_rate']:.1f}% "
            f"({metrics['wins']}/{metrics['trades']}) | Max DD: {metrics['max_dd']:.2f}%",
            "<b>Volume &amp; 20-Period Moving Average</b>"
        )
    )

    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # 2. VWAP (if available)
    if "vwap" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["vwap"],
            mode="lines", name="Session VWAP",
            line=dict(color="#f39c12", width=1.5, dash="dot"), opacity=0.85
        ), row=1, col=1)

    # 3. Volume
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(plot_df["close"], plot_df["open"])
    ]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["volume"],
        name="Volume", marker_color=vol_colors, opacity=0.7
    ), row=2, col=1)

    if "vol_ma" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["vol_ma"],
            mode="lines", name="Vol MA(20)",
            line=dict(color="#ff9800", width=1.2)
        ), row=2, col=1)

    # 4. Trade overlays
    for t in trades:
        et = t.get("entry_time")
        xt = t.get("exit_time")
        ep = t.get("entry_price")
        xp = t.get("exit_price")
        sl = t.get("stop_loss")
        tp = t.get("take_profit")
        side = t.get("side", "buy")
        pnl = t.get("pnl", 0)
        is_win = pnl > 0

        if not (et and ep):
            continue

        marker_color = "#26a69a" if side == "buy" else "#ef5350"
        marker_sym = "triangle-up" if side == "buy" else "triangle-down"

        fig.add_trace(go.Scatter(
            x=[et], y=[ep],
            mode="markers+text",
            marker=dict(symbol=marker_sym, size=12, color=marker_color, line=dict(color="#ffffff", width=1)),
            text=[f"{'BUY' if side == 'buy' else 'SELL'} @{ep:.2f}"],
            textposition="top center" if side == "buy" else "bottom center",
            textfont=dict(size=9, color=marker_color),
            name=f"Entry ({side.upper()})",
            showlegend=False,
        ), row=1, col=1)

        if xp and xt:
            exit_color = "#26a69a" if is_win else "#ef5350"
            fig.add_trace(go.Scatter(
                x=[xt], y=[xp],
                mode="markers+text",
                marker=dict(symbol="x", size=10, color=exit_color, line=dict(color="#ffffff", width=1)),
                text=[f"EXIT @{xp:.2f} ({'+' if pnl >= 0 else ''}{pnl:.2f})"],
                textposition="top right",
                textfont=dict(size=9, color=exit_color),
                name=f"Exit ({'Win' if is_win else 'Loss'})",
                showlegend=False,
            ), row=1, col=1)

            # Draw trade line
            fig.add_trace(go.Scatter(
                x=[et, xt], y=[ep, xp],
                mode="lines",
                line=dict(color=exit_color, width=1, dash="dot"),
                showlegend=False
            ), row=1, col=1)

        if sl:
            fig.add_trace(go.Scatter(
                x=[et, xt or et], y=[sl, sl],
                mode="lines",
                line=dict(color="#ef5350", width=1, dash="longdash"),
                name="Stop Loss", showlegend=False
            ), row=1, col=1)

        if tp:
            fig.add_trace(go.Scatter(
                x=[et, xt or et], y=[tp, tp],
                mode="lines",
                line=dict(color="#26a69a", width=1, dash="dash"),
                name="Take Profit", showlegend=False
            ), row=1, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        margin=dict(l=60, r=60, t=80, b=40),
        height=700,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="#1e222d", bordercolor="#2a2e39", borderwidth=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="#1e222d", showgrid=True, zeroline=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[20, 9.5], pattern="hour"),
            ]
        ),
        xaxis2=dict(gridcolor="#1e222d", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1e222d", showgrid=True, side="right"),
        yaxis2=dict(gridcolor="#1e222d", showgrid=True, side="right"),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return fig


# ── Helper: Run one backtest ───────────────────────────────────────────────────
def run_one_backtest(ticker, strategy_cfg, bt_start, bt_end, data_start, budget_val, log_cb):
    """Runs a single backtest and returns (metrics dict, plotly fig, chart_path)."""
    strat_cls = strategy_cfg["cls"]
    timeframe = strategy_cfg["timeframe"]
    datasource = strategy_cfg["datasource"]
    rth_only = strategy_cfg["rth_only"]

    log_cb(f"  Loading data for {ticker} ({timeframe})...")

    asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)

    if datasource in ["mysql", "multitf"]:
        # Try local MySQL institutional candle feed first
        df_raw = candle_feed.get_candles_df(
            ticker,
            timeframe=timeframe,
            start_date=data_start,
            end_date=bt_end.strftime("%Y-%m-%d %H:%M:%S"),
            rth_only=rth_only
        )
        if df_raw.empty or len(df_raw) < 50:
            log_cb(f"  Local MySQL has no 5m data for {ticker}. Fetching from Yahoo Finance...")
            import yfinance as yf
            from lumibot.entities import Data
            df_yf = yf.download(ticker, start=data_start, end=bt_end.strftime("%Y-%m-%d"), interval="5m", progress=False)
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.droplevel(1)
            df_yf.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            df_raw = df_yf.copy()
            if df_raw.empty or len(df_raw) < 50:
                raise ValueError(f"Insufficient 5-min data from Yahoo for {ticker} ({len(df_raw)} bars)")
            log_cb(f"  Loaded {len(df_raw):,} 5-min Yahoo bars.")
            pandas_data = {asset: Data(asset, df_raw, timestep="minute")}
        else:
            log_cb(f"  Loaded {len(df_raw):,} 5-minute RTH candles from MySQL.")
            pandas_data = candle_feed.create_lumibot_pandas_data(
                ticker,
                timeframe=timeframe,
                start_date=data_start,
                end_date=bt_end.strftime("%Y-%m-%d %H:%M:%S"),
                rth_only=rth_only
            )

        log_cb(f"  Running backtest on {ticker}...")
        results, strategy_obj = strat_cls.run_backtest(
            datasource_class=PandasDataBacktesting,
            backtesting_start=bt_start,
            backtesting_end=bt_end,
            pandas_data=pandas_data,
            budget=budget_val,
            parameters={"Ticker": asset, "Plot": False},
            show_plot=False
        )
    elif datasource == "yahoo":
        log_cb(f"  Fetching daily data from Yahoo Finance for {ticker}...")
        results, strategy_obj = strat_cls.run_backtest(
            datasource_class=YahooDataBacktesting,
            backtesting_start=bt_start,
            backtesting_end=bt_end,
            budget=budget_val,
            parameters={"Ticker": asset, "Plot": False},
            show_plot=False
        )
        df_raw = None
    else:
        raise ValueError(f"Unknown datasource: {datasource}")

    # Extract metrics from Lumibot results
    tot_return = results.get("total_return", 0.0) * 100.0
    mdd_raw = results.get("max_drawdown", 0.0)
    mdd = (mdd_raw.get("drawdown", 0.0) if isinstance(mdd_raw, dict) else mdd_raw) * 100.0

    # ── Normalise trade records ──────────────────────────────────────────────
    raw_trades = getattr(strategy_obj, "trade_records", None)
    tracker = getattr(strategy_obj, "trade_tracker", None)

    if raw_trades is not None:
        # Already normalised dicts
        trades = raw_trades
    elif tracker is not None:
        # Convert Trade dataclass → standard dict
        trades = []
        for t in tracker.closed_trades:
            trades.append({
                "entry_time":  str(t.date_executed),
                "exit_time":   str(t.date_completed) if t.date_completed else None,
                "side":        "buy" if t.trade_type.upper() == "BUY" else "sell",
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "stop_loss":   t.stop_loss,
                "take_profit": t.take_profit,
                "pnl":         t.pnl if t.pnl is not None else 0.0,
                "pnl_pct":     t.pnl_percent if t.pnl_percent is not None else 0.0,
            })
    else:
        trades = []

    total_trades = len(trades)
    wins = [t for t in trades if t.get("pnl", 0.0) > 0]
    losses = [t for t in trades if t.get("pnl", 0.0) < 0]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = sum(t.get("pnl", 0.0) for t in trades)

    metrics = {
        "ticker": ticker,
        "return_pct": tot_return,
        "max_dd": mdd,
        "trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "final_equity": budget_val + total_pnl,
    }

    log_cb(
        f"  {ticker} → Return: {tot_return:+.2f}% | "
        f"Win Rate: {win_rate:.1f}% ({len(wins)}W/{len(losses)}L) | "
        f"Trades: {total_trades} | PnL: ${total_pnl:+.2f}"
    )

    # Chart
    strategy_slug = strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    chart_file = str(CHARTS_DIR / f"{ticker}_{strategy_slug}.html")

    # If intraday strategy, compute indicators from df_raw for chart enrichment
    if df_raw is not None and hasattr(strategy_obj, "_compute_intraday_indicators"):
        plot_df = strategy_obj._compute_intraday_indicators(df_raw)
    elif df_raw is not None:
        plot_df = df_raw.copy()
    else:
        # Yahoo-backed — get_historical_prices isn't available post-backtest;
        # reconstruct a minimal df from the equity curve if possible
        plot_df = pd.DataFrame()

    if not plot_df.empty:
        fig = generate_plotly_chart(ticker, plot_df, trades, metrics, chart_file)
    else:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#131722",
            title=f"{ticker} — No OHLCV data available for charting (Yahoo daily strategy)"
        )
        fig.write_html(chart_file, include_plotlyjs="cdn")

    metrics["chart_path"] = chart_file
    return metrics, fig, chart_file


# ── Helper: Generate unified dashboard HTML ────────────────────────────────────
def generate_dashboard(all_metrics, tickers_list):
    slug = strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    dash_path = CHARTS_DIR / f"dashboard_{slug}.html"
    rows_html = ""
    buttons_html = ""
    first_chart = ""

    for i, m in enumerate(all_metrics):
        if not m or "return_pct" not in m:
            continue
        rc = "#26a69a" if m["return_pct"] >= 0 else "#ef5350"
        pc = "#26a69a" if m["total_pnl"] >= 0 else "#ef5350"
        wc = "#26a69a" if m["win_rate"] >= 50 else "#f39c12"
        chart_name = Path(m.get("chart_path", "")).name
        if i == 0:
            first_chart = chart_name
        rows_html += f"""
        <tr>
            <td><a href="{chart_name}" target="cf" onclick="sw('{chart_name}','{m['ticker']}')" style="color:#29b6f6;text-decoration:none;font-weight:700;">{m['ticker']}</a></td>
            <td style="color:{rc};font-weight:700;">{m['return_pct']:+.2f}%</td>
            <td style="color:#ef5350;">{m['max_dd']:.2f}%</td>
            <td style="color:{wc};font-weight:700;">{m['win_rate']:.1f}% ({m['wins']}W / {m['losses']}L)</td>
            <td>{m['trades']}</td>
            <td style="color:{pc};font-weight:700;">${m['total_pnl']:+,.2f}</td>
            <td>${m['final_equity']:,.2f}</td>
            <td><a href="{chart_name}" target="_blank" style="background:#1e88e5;color:#fff;padding:4px 10px;border-radius:4px;text-decoration:none;font-size:0.85em;">Open ↗</a></td>
        </tr>"""
        act = " class='active'" if i == 0 else ""
        buttons_html += f"<button id='btn-{m['ticker']}'{act} onclick=\"sw('{chart_name}','{m['ticker']}')\">{m['ticker']}</button>"

    html = f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><title>{strategy_name} — Gotti Dashboard</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#131722;color:#d1d4dc;margin:0;padding:20px;}}
h1{{color:#fff;margin:0 0 4px;font-size:1.5em;}}
.sub{{color:#787b86;font-size:0.88em;margin-bottom:20px;}}
table{{width:100%;border-collapse:collapse;background:#1e222d;border:1px solid #2a2e39;border-radius:8px;overflow:hidden;margin-bottom:24px;}}
th,td{{padding:12px 16px;text-align:left;border-bottom:1px solid #2a2e39;}}
th{{background:#2a2e39;font-size:0.82em;text-transform:uppercase;letter-spacing:.5px;color:#9598a1;}}
tr:hover td{{background:#252a36;}}
.chart-box{{background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:16px;}}
.btn-row{{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;}}
.btn-row button{{background:#2a2e39;border:1px solid #363c4e;color:#d1d4dc;padding:6px 16px;border-radius:4px;cursor:pointer;font-weight:600;font-size:.9em;}}
.btn-row button.active{{background:#1e88e5;color:#fff;border-color:#1e88e5;}}
iframe{{width:100%;height:780px;border:none;border-radius:6px;}}
</style></head>
<body>
<h1>📈 {strategy_name}</h1>
<div class="sub">Window: {start_date} → {end_date} &nbsp;|&nbsp; Starting Budget: ${budget_val:,.0f} per ticker &nbsp;|&nbsp; Timeframe: {cfg['timeframe']}</div>
<table>
<thead><tr><th>Ticker</th><th>Total Return</th><th>Max Drawdown</th><th>Win Rate</th><th>Trades</th><th>Net PnL</th><th>Final Equity</th><th>Standalone</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<div class="chart-box">
<div class="btn-row" id="btn-row">{buttons_html}</div>
<iframe id="cf" name="cf" src="{first_chart}"></iframe>
</div>
<script>
function sw(url,t){{document.getElementById('cf').src=url;document.querySelectorAll('.btn-row button').forEach(b=>b.classList.remove('active'));var b=document.getElementById('btn-'+t);if(b)b.classList.add('active');}}
</script>
</body></html>"""

    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html)
    return str(dash_path)


# ── Backtest Execution Block ───────────────────────────────────────────────────
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "results" not in st.session_state:
    st.session_state.results = []
if "figs" not in st.session_state:
    st.session_state.figs = {}
if "dashboard_path" not in st.session_state:
    st.session_state.dashboard_path = None
if "running" not in st.session_state:
    st.session_state.running = False

if run_button:
    if not tickers:
        st.error("Please enter at least one ticker symbol.")
    else:
        st.session_state.log_lines = []
        st.session_state.results = []
        st.session_state.figs = {}
        st.session_state.dashboard_path = None
        st.session_state.running = True

        bt_start = datetime.combine(start_date, datetime.min.time())
        bt_end = datetime.combine(end_date, datetime.min.time())
        data_start = (start_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        budget_val = budget

        log_placeholder = st.empty()
        status_placeholder = st.empty()

        def log_cb(msg):
            st.session_state.log_lines.append(msg)
            log_placeholder.markdown(
                f"<div class='log-box'>{'<br>'.join(st.session_state.log_lines[-50:])}</div>",
                unsafe_allow_html=True
            )

        all_metrics = []
        all_figs = {}

        for ticker in tickers:
            log_cb(f"\n{'='*55}")
            log_cb(f"  ▶  {ticker}  ·  {strategy_name}")
            log_cb(f"{'='*55}")
            try:
                metrics, fig, chart_path = run_one_backtest(
                    ticker=ticker,
                    strategy_cfg=cfg,
                    bt_start=bt_start,
                    bt_end=bt_end,
                    data_start=data_start,
                    budget_val=budget_val,
                    log_cb=log_cb
                )
                all_metrics.append(metrics)
                all_figs[ticker] = (fig, chart_path)
                log_cb(f"  ✓ Chart saved → {Path(chart_path).name}")
            except Exception as e:
                log_cb(f"  ✗ ERROR on {ticker}: {e}")
                import traceback
                log_cb(traceback.format_exc())

        # Generate dashboard
        if all_metrics:
            try:
                dash_path = generate_dashboard(all_metrics, tickers)
                st.session_state.dashboard_path = dash_path
                log_cb(f"\n  ✓ Dashboard saved → {Path(dash_path).name}")
            except Exception as e:
                log_cb(f"  ✗ Dashboard error: {e}")

        log_cb("\n  ✅ Backtest run complete.")
        st.session_state.results = all_metrics
        st.session_state.figs = all_figs
        st.session_state.running = False

# ── Display Results ────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    valid = [m for m in results if m and "return_pct" in m]

    st.markdown("---")
    st.markdown("### 📊 Performance Summary")

    # Metric tiles row
    cols = st.columns(len(valid) if len(valid) <= 4 else 4)
    for i, m in enumerate(valid):
        col = cols[i % len(cols)]
        ret = m["return_pct"]
        cls_col = "pos" if ret >= 0 else "neg"
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{m['ticker']}</div>"
            f"<div class='value {cls_col}'>{ret:+.2f}%</div>"
            f"<div style='color:#787b86;font-size:0.82em;margin-top:4px;'>"
            f"WR {m['win_rate']:.1f}% · {m['trades']} trades · ${m['total_pnl']:+,.0f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Full results table
    table_rows = ""
    for m in valid:
        rc = "#26a69a" if m["return_pct"] >= 0 else "#ef5350"
        pc = "#26a69a" if m["total_pnl"] >= 0 else "#ef5350"
        wc = "#26a69a" if m["win_rate"] >= 50 else "#f39c12"
        chart_link = ""
        if "chart_path" in m:
            chart_link = f'<a href="file:///{m["chart_path"]}" target="_blank" style="color:#29b6f6;">Open Chart ↗</a>'
        table_rows += f"""
        <tr>
            <td><b>{m['ticker']}</b></td>
            <td style="color:{rc};font-weight:700;">{m['return_pct']:+.2f}%</td>
            <td style="color:#ef5350;">{m['max_dd']:.2f}%</td>
            <td style="color:{wc};font-weight:700;">{m['win_rate']:.1f}% ({m['wins']}W / {m['losses']}L)</td>
            <td>{m['trades']}</td>
            <td style="color:{pc};font-weight:700;">${m['total_pnl']:+,.2f}</td>
            <td>${m['final_equity']:,.2f}</td>
            <td>{chart_link}</td>
        </tr>"""

    st.markdown(f"""
    <table class="result-table">
    <thead><tr>
        <th>Ticker</th><th>Total Return</th><th>Max Drawdown</th>
        <th>Win Rate</th><th>Trades</th><th>Net PnL</th><th>Final Equity</th><th>Chart</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Dashboard link
    if st.session_state.dashboard_path:
        dash = st.session_state.dashboard_path
        st.markdown(
            f"<div style='background:#1e222d;border:1px solid #2a2e39;border-radius:8px;padding:16px;margin-bottom:20px;'>"
            f"<b style='color:#f39c12;'>📊 Unified Dashboard</b> "
            f"— Open <a href='file:///{dash}' target='_blank' style='color:#29b6f6;'>{Path(dash).name}</a> "
            f"in your browser to view all charts in one tabbed interface."
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📈 Interactive Charts")

    # Ticker tab selector
    if st.session_state.figs:
        tab_labels = [m["ticker"] for m in valid if m["ticker"] in st.session_state.figs]
        if tab_labels:
            tabs = st.tabs(tab_labels)
            for tab, ticker in zip(tabs, tab_labels):
                fig_obj, chart_path = st.session_state.figs[ticker]
                with tab:
                    st.plotly_chart(fig_obj, use_container_width=True, key=f"chart_{ticker}")
                    st.markdown(
                        f"<a href='file:///{chart_path}' target='_blank' "
                        f"style='color:#29b6f6;font-size:0.88em;'>🔗 Open standalone HTML chart in browser</a>",
                        unsafe_allow_html=True
                    )

# ── Empty State ────────────────────────────────────────────────────────────────
elif not st.session_state.running:
    st.markdown("""
    <div style='background:#1e222d;border:1px dashed #2a2e39;border-radius:12px;padding:48px;text-align:center;margin-top:20px;'>
        <div style='font-size:3em;'>📊</div>
        <div style='color:#ffffff;font-size:1.2em;font-weight:700;margin-top:12px;'>No backtest results yet</div>
        <div style='color:#787b86;margin-top:8px;'>
            Configure your strategy, tickers, and date range in the sidebar.<br>
            Click <b style='color:#1e88e5;'>🚀 Run Backtest</b> to start.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show existing charts hint
    existing = sorted(CHARTS_DIR.glob("*.html"))
    if existing:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Previously generated charts:**")
        for f in existing[-8:]:
            st.markdown(f"🔗 [`{f.name}`](file:///{f})")
