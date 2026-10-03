"""
Institutional Liquidity Sweep Multi-Asset Simulation Runner
============================================================
Simulates the Institutional Liquidity Sweep strategy on 5-minute candles
over a 3-month window (2026-06-15 to 2026-09-17) for:
  - NVDA (Nvidia)
  - PLTR (Palantir)
  - MARA (MARA Holdings)
  - GLD  (SPDR Gold Trust)

Generates interactive institutional Plotly charts saved to logs/charts/
"""

import sys
import os
import io
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Disable Lumibot progress bar to prevent Windows cp1252 encoding crashes
import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None
import lumibot.data_sources.data_source_backtesting
lumibot.data_sources.data_source_backtesting.print_progress_bar = lambda *args, **kwargs: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.entities import Asset
from lumibot.backtesting import PandasDataBacktesting
from plots.mysql_candle_feed import candle_feed
from strategies.liquidity_sweep import LiquiditySweepStrategy


TICKERS = ["NVDA", "PLTR", "MARA", "GLD"]
DATA_START = "2026-06-01"  # Buffer for lookbacks and initial VWAP/pivots
BACKTEST_START = datetime(2026, 6, 15)
BACKTEST_END = datetime(2026, 9, 17)
STARTING_BUDGET = 10_000.0


def generate_interactive_chart(ticker: str, df: pd.DataFrame, trades: list, metrics: dict, output_path: str):
    """
    Builds a full institutional multi-panel candlestick chart with VWAP,
    trade entries, exits, stop loss + ATR buffers, and volume.
    """
    # Filter df to backtest window
    plot_df = df[(df.index >= "2026-06-15") & (df.index <= "2026-09-18")].copy()
    if plot_df.empty:
        plot_df = df.copy()

    # Create 2-row subplot (Price + Volume)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(
            f"<b>{ticker}</b> 5-Min Institutional Liquidity Sweep | Return: {metrics['return_pct']:.2f}% | Win Rate: {metrics['win_rate']:.1f}% ({metrics['wins']}/{metrics['trades']}) | Max DD: {metrics['max_dd']:.2f}%",
            "<b>Volume & 20-Period Moving Average</b>"
        )
    )

    # 1. Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="5M Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1, col=1
    )

    # 2. Session VWAP
    if "vwap" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["vwap"],
                mode="lines",
                name="Session VWAP",
                line=dict(color="#f39c12", width=1.5, dash="dot"),
                opacity=0.85
            ),
            row=1, col=1
        )

    # 3. Volume Bar Chart
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(plot_df["close"], plot_df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    if "vol_ma" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["vol_ma"],
                mode="lines",
                name="Volume MA(20)",
                line=dict(color="#3498db", width=1.2),
            ),
            row=2, col=1
        )

    # 4. Plot Trades
    long_entries_x, long_entries_y, long_text = [], [], []
    short_entries_x, short_entries_y, short_text = [], [], []
    win_exits_x, win_exits_y, win_text = [], [], []
    loss_exits_x, loss_exits_y, loss_text = [], [], []

    for t in trades:
        entry_t = pd.to_datetime(t.get("entry_time"))
        exit_t = pd.to_datetime(t.get("exit_time")) if t.get("exit_time") else None
        side = t.get("side", "buy")
        entry_p = t.get("entry_price")
        exit_p = t.get("exit_price")
        pnl = t.get("pnl", 0.0)
        pnl_pct = t.get("pnl_pct", 0.0)
        sl = t.get("stop_loss")
        tp = t.get("take_profit")
        lvl_name = t.get("level_name", "Clean Level")
        extreme = t.get("sweep_extreme")

        # Entry point
        hover_info = (
            f"<b>{side.upper()} ENTRY</b><br>"
            f"Time: {entry_t}<br>"
            f"Price: ${entry_p:.2f}<br>"
            f"Swept: {lvl_name} (${t.get('target_level', 0):.2f})<br>"
            f"Sweep Wick: ${extreme:.2f}<br>"
            f"Stop Loss (w/ ATR Buffer): ${sl:.2f}<br>"
            f"Take Profit: ${tp:.2f}<br>"
            f"Shares: {t.get('quantity')}"
        )

        if side == "buy":
            long_entries_x.append(entry_t)
            long_entries_y.append(entry_p)
            long_text.append(hover_info)
        else:
            short_entries_x.append(entry_t)
            short_entries_y.append(entry_p)
            short_text.append(hover_info)

        # Exit point and connection line
        if exit_t and exit_p:
            exit_hover = (
                f"<b>{'WIN' if pnl >= 0 else 'LOSS'} EXIT</b><br>"
                f"Exit Time: {exit_t}<br>"
                f"Exit Price: ${exit_p:.2f}<br>"
                f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
            )
            if pnl >= 0:
                win_exits_x.append(exit_t)
                win_exits_y.append(exit_p)
                win_text.append(exit_hover)
            else:
                loss_exits_x.append(exit_t)
                loss_exits_y.append(exit_p)
                loss_text.append(exit_hover)

            # Draw trade bracket projection (Stop Loss & Take Profit levels)
            fig.add_trace(
                go.Scatter(
                    x=[entry_t, exit_t],
                    y=[tp, tp],
                    mode="lines",
                    line=dict(color="#2ecc71", width=1.5, dash="dash"),
                    name="Take Profit Level",
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[entry_t, exit_t],
                    y=[sl, sl],
                    mode="lines",
                    line=dict(color="#e74c3c", width=1.5, dash="dash"),
                    name="Stop Loss Level (w/ ATR)",
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=1, col=1
            )
            # Connecting trade path
            fig.add_trace(
                go.Scatter(
                    x=[entry_t, exit_t],
                    y=[entry_p, exit_p],
                    mode="lines",
                    line=dict(color="#2ecc71" if pnl >= 0 else "#e74c3c", width=1.8),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=1, col=1
            )

    # Add Entry Scatters
    if long_entries_x:
        fig.add_trace(
            go.Scatter(
                x=long_entries_x,
                y=long_entries_y,
                mode="markers",
                name="Long Entry",
                marker=dict(symbol="triangle-up", size=13, color="#2ecc71", line=dict(color="#ffffff", width=1)),
                text=long_text,
                hoverinfo="text"
            ),
            row=1, col=1
        )
    if short_entries_x:
        fig.add_trace(
            go.Scatter(
                x=short_entries_x,
                y=short_entries_y,
                mode="markers",
                name="Short Entry",
                marker=dict(symbol="triangle-down", size=13, color="#e74c3c", line=dict(color="#ffffff", width=1)),
                text=short_text,
                hoverinfo="text"
            ),
            row=1, col=1
        )

    # Add Exit Scatters
    if win_exits_x:
        fig.add_trace(
            go.Scatter(
                x=win_exits_x,
                y=win_exits_y,
                mode="markers",
                name="Win Exit",
                marker=dict(symbol="circle", size=11, color="#2ecc71", line=dict(color="#ffffff", width=1.5)),
                text=win_text,
                hoverinfo="text"
            ),
            row=1, col=1
        )
    if loss_exits_x:
        fig.add_trace(
            go.Scatter(
                x=loss_exits_x,
                y=loss_exits_y,
                mode="markers",
                name="Loss Exit",
                marker=dict(symbol="x", size=11, color="#e74c3c", line=dict(color="#ffffff", width=1.5)),
                text=loss_text,
                hoverinfo="text"
            ),
            row=1, col=1
        )

    # Styling and Layout
    fig.update_layout(
        template="plotly_dark",
        height=880,
        margin=dict(l=50, r=50, t=70, b=40),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor="#11151c",
        plot_bgcolor="#161b22",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  -> Saved chart to: {output_path}")


def run_simulation_for_ticker(ticker: str) -> dict:
    """Executes a 3-month 5-minute backtest and generates the interactive chart."""
    print(f"\n=======================================================")
    print(f"  Simulating Liquidity Sweep on {ticker} (3-Month 5M)")
    print(f"=======================================================")

    # 1. Fetch 5-Minute Historical Candles
    df_raw = candle_feed.get_candles_df(
        ticker,
        timeframe="5 min",
        start_date=DATA_START,
        end_date=BACKTEST_END.strftime("%Y-%m-%d %H:%M:%S")
    )

    if df_raw.empty or len(df_raw) < 100:
        print(f"  [ERROR] Insufficient candles found for {ticker}")
        return {"ticker": ticker, "status": "ERROR"}

    print(f"  Loaded {len(df_raw):,} 5-minute institutional candles.")

    # 2. Package for Lumibot
    pandas_data = candle_feed.create_lumibot_pandas_data(
        ticker,
        timeframe="5 min",
        start_date=DATA_START,
        end_date=BACKTEST_END.strftime("%Y-%m-%d %H:%M:%S")
    )

    # 3. Run Lumibot Backtest
    asset = Asset(symbol=ticker.upper(), asset_type=Asset.AssetType.STOCK)
    results, strategy = LiquiditySweepStrategy.run_backtest(
        datasource_class=PandasDataBacktesting,
        backtesting_start=BACKTEST_START,
        backtesting_end=BACKTEST_END,
        pandas_data=pandas_data,
        budget=STARTING_BUDGET,
        parameters={"Ticker": asset, "Plot": False},
        show_plot=False
    )

    # 4. Extract Performance Metrics
    tot_return = results.get("total_return", 0.0) * 100.0
    mdd_raw = results.get("max_drawdown", 0.0)
    mdd = (mdd_raw.get("drawdown", 0.0) if isinstance(mdd_raw, dict) else mdd_raw) * 100.0

    trades = strategy.trade_records
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.get("pnl", 0.0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0.0) < 0]
    win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = sum([t.get("pnl", 0.0) for t in trades])

    metrics = {
        "ticker": ticker,
        "return_pct": tot_return,
        "max_dd": mdd,
        "trades": total_trades,
        "wins": len(winning_trades),
        "losses": len(losing_trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "final_equity": STARTING_BUDGET + total_pnl,
    }

    print(f"  Total Trades:     {total_trades}")
    print(f"  Win Rate:         {win_rate:.1f}% ({len(winning_trades)} wins / {len(losing_trades)} losses)")
    print(f"  Total Return:     {tot_return:+.2f}%")
    print(f"  Max Drawdown:     {mdd:.2f}%")
    print(f"  Net Strategy PnL: ${total_pnl:+.2f}")

    # 5. Compute indicators for chart visualization
    plot_df = strategy._compute_intraday_indicators(df_raw)

    # 6. Generate Chart
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    chart_file = os.path.join(repo_root, "logs", "charts", f"{ticker}_liquidity_sweep_5m.html")
    generate_interactive_chart(ticker, plot_df, trades, metrics, chart_file)
    metrics["chart_path"] = chart_file

    return metrics


def main():
    print("=" * 65)
    print("  INSTITUTIONAL LIQUIDITY SWEEP: 3-MONTH MULTI-ASSET SIMULATION")
    print("  Timeframe: 5-Minute Candles | Model: Sweep + Reclaim + Structure Shift")
    print(f"  Window:    {BACKTEST_START.strftime('%Y-%m-%d')} to {BACKTEST_END.strftime('%Y-%m-%d')}")
    print("=" * 65)

    all_metrics = []
    for ticker in TICKERS:
        try:
            m = run_simulation_for_ticker(ticker)
            all_metrics.append(m)
        except Exception as e:
            print(f"  [ERROR] Failed simulation on {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 75)
    print("  FINAL 3-MONTH SIMULATION PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f"{'Ticker':<8} | {'Return (%)':<12} | {'Max DD (%)':<12} | {'Win Rate (%)':<14} | {'Trades':<8} | {'Net PnL ($)':<12}")
    print("-" * 75)
    for m in all_metrics:
        if "return_pct" in m:
            print(f"{m['ticker']:<8} | {m['return_pct']:>+10.2f}% | {m['max_dd']:>10.2f}% | {m['win_rate']:>11.1f}% | {m['trades']:>6} | ${m['total_pnl']:>+10.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
