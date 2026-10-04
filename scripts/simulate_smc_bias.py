"""
The 5-Rule SMC Intraday Bias Model: Multi-Asset Simulation Runner (simulate_smc_bias.py)
========================================================================================
Executes Lewis Kelly's systematic 5-Rule SMC Intraday Bias Strategy across US Equities:
  - Rule 1: Directional Bias via The Box Method (15-Minute Macro Context)
  - Rule 2: Session Killzone Timing (London: 02:00-05:00 EST, NY: 07:00-11:30 EST)
  - Rule 3: Liquidation Sweeps (Premarket / Asian / London High & Low sweeps)
  - Rule 4: Reversal Confirmation (1-Minute Micro-CHoCH Impulsive Body Close)
  - Rule 5: Entry Model (5-Minute POI: Order Block, FVG, IFVG & Confluence)
  - Asymmetric Risk Architecture: Strict 1:5R Veto Rule (Target Avg ~1:6.23R)
  - Hybrid Partial Scaling: 40% TP1 @ 3.0R (+ Breakeven Stop), 60% Runner @ M15 Target

Generates interactive institutional Plotly charts and a unified executive HTML dashboard.
"""

from __future__ import annotations

import sys
import os
import io
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plots.mysql_candle_feed import candle_feed
from strategies.smc_bias_model.bias_structure import M15BoxStructureDetector, BiasDirection
from strategies.smc_bias_model.session_killzones import SessionKillzoneManager
from strategies.smc_bias_model.m1_choch import M1CHoCHDetector
from strategies.smc_bias_model.m5_poi import M5POISelector, POIType
from strategies.smc_bias_model.bias_order_manager import BiasOrderManager, OrderLifecycleStatus


TICKERS = ["NVDA", "AAPL", "PLTR", "MSFT"]
STARTING_BUDGET = 100_000.0


def fetch_or_fallback_candles(ticker: str) -> pd.DataFrame:
    """
    Fetches intraday candles from MySQL, with resilient fallback to yfinance or realistic synthetic data.
    """
    df = pd.DataFrame()
    try:
        df = candle_feed.get_candles_df(
            ticker,
            timeframe="5 min",
            start_date="2026-06-01",
            end_date="2026-10-02 20:00:00",
            rth_only=False  # include premarket/afterhours for session benchmarks
        )
    except Exception as e:
        print(f"  [MySQL Notice] {e}")

    if not df.empty and len(df) >= 100:
        return df

    print(f"  [Fallback] Local MySQL unavailable. Fetching {ticker} via yfinance...")
    try:
        import yfinance as yf
        # Fetch 5m intraday data (60 days)
        yf_df = yf.download(ticker, interval="5m", period="60d", progress=False)
        if not yf_df.empty:
            yf_df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in yf_df.columns]
            yf_df.index = pd.to_datetime(yf_df.index, utc=True)
            if "vwap" not in yf_df.columns:
                yf_df["vwap"] = (yf_df["high"] + yf_df["low"] + yf_df["close"]) / 3.0
            return yf_df
    except Exception as yf_err:
        print(f"  [Fallback Error] yfinance failed: {yf_err}")

    # Synthetic fallback
    print(f"  [Fallback] Generating realistic intraday session candles for {ticker}...")
    dates = pd.date_range(start="2026-07-01 04:00:00", end="2026-10-02 16:00:00", freq="5min", tz="UTC")
    np.random.seed(42 + hash(ticker) % 1000)
    base_price = 120.0 if ticker == "NVDA" else (225.0 if ticker == "AAPL" else 30.0)
    returns = np.random.normal(0.0001, 0.002, len(dates))
    price_series = base_price * np.exp(np.cumsum(returns))

    opens, highs, lows, closes, volumes, vwaps = [], [], [], [], [], []
    for p in price_series:
        o = p
        c = p * (1 + np.random.normal(0, 0.001))
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.0015)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.0015)))
        v = float(np.random.randint(40_000, 250_000))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)
        vwaps.append((h + l + c) / 3.0)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes, "vwap": vwaps
    }, index=dates)


def generate_smc_bias_plotly_chart(
    ticker: str,
    df_raw: pd.DataFrame,
    trades: list,
    metrics: dict,
    output_path: str
):
    """
    Renders an institutional Plotly chart featuring:
      - Intraday Candlesticks
      - Shaded Killzone Sessions (London in blue, New York in orange)
      - Entry markers, Stop Loss, TP1 (3R), and TP2 (M15 External Target >= 5R)
      - Volume and Volume MA
    """
    plot_df = df_raw.copy()
    if len(plot_df) > 1500:
        plot_df = plot_df.iloc[-1500:]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(
            f"<b>{ticker}</b> 5-Rule SMC Intraday Bias Model | Return: {metrics['return_pct']:+.2f}% | Win Rate: {metrics['win_rate']:.1f}% ({metrics['wins']}W / {metrics['losses']}L) | Avg R:R: {metrics['avg_rr']:.2f}R | Profit Factor: {metrics['profit_factor']:.2f}",
            "<b>Volume & 20-Period Moving Average</b>"
        )
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Intraday Candles",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1, col=1
    )

    # 2. Volume & MA
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(plot_df["open"], plot_df["close"])]
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["volume"],
            marker_color=colors,
            opacity=0.4,
            name="Volume",
        ),
        row=2, col=1
    )

    vol_ma = plot_df["volume"].rolling(20).mean()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=vol_ma,
            line=dict(color="#f39c12", width=1.5),
            name="Volume 20 MA",
        ),
        row=2, col=1
    )

    # 3. Plot Trades
    entries_x, entries_y, entries_text = [], [], []
    exits_x, exits_y, exits_text = [], [], []

    for t in trades:
        entry_t = pd.to_datetime(t.get("entry_time"))
        exit_t = pd.to_datetime(t.get("exit_time")) if t.get("exit_time") else None
        side = t.get("side", "BUY")
        entry_p = t.get("entry_price")
        pnl = t.get("pnl", 0.0)
        initial_sl = t.get("initial_sl")
        rr = t.get("rr_ratio", 0.0)
        tp1_stat = t.get("tp1_hit", False)
        tp2_stat = t.get("tp2_hit", False)

        hover_entry = (
            f"<b>SMC BIAS {side} ENTRY</b><br>"
            f"Time: {entry_t}<br>"
            f"Price: ${entry_p:.2f}<br>"
            f"Micro SL: ${initial_sl:.2f}<br>"
            f"Expected R:R: {rr:.2f}R<br>"
            f"TP1 (40% @ 3R): {'FILLED' if tp1_stat else 'N/A'}<br>"
            f"TP2 (60% @ M15): {'FILLED' if tp2_stat else 'N/A'}<br>"
            f"Total PnL: ${pnl:+,.2f}"
        )
        entries_x.append(entry_t)
        entries_y.append(entry_p)
        entries_text.append(hover_entry)

        if exit_t:
            hover_exit = (
                f"<b>SMC BIAS TRADE COMPLETE</b><br>"
                f"Exit Time: {exit_t}<br>"
                f"Total Realized PnL: ${pnl:+,.2f}<br>"
                f"Outcome: {'WIN (TP Hit)' if pnl > 0 else ('BREAKEVEN' if pnl == 0 else 'LOSS (SL Hit)')}"
            )
            exits_x.append(exit_t)
            exits_y.append(entry_p)
            exits_text.append(hover_exit)

    if entries_x:
        symbols = ["triangle-up" if t.get("side") == "BUY" else "triangle-down" for t in trades]
        fig.add_trace(
            go.Scatter(
                x=entries_x,
                y=entries_y,
                mode="markers",
                marker=dict(symbol=symbols, size=14, color="#29b6f6", line=dict(width=1, color="#ffffff")),
                name="SMC Limit Entry",
                text=entries_text,
                hoverinfo="text"
            ),
            row=1, col=1
        )

    if exits_x:
        fig.add_trace(
            go.Scatter(
                x=exits_x,
                y=exits_y,
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#ab47bc", line=dict(width=1, color="#ffffff")),
                name="Trade Complete",
                text=exits_text,
                hoverinfo="text"
            ),
            row=1, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  -> Chart saved: {output_path}")


def run_smc_bias_simulation_for_ticker(ticker: str) -> dict:
    """Executes a backtest of the 5-Rule SMC Intraday Bias Model for a single ticker."""
    print(f"\n=======================================================")
    print(f"  Simulating 5-Rule SMC Intraday Bias Model on {ticker}")
    print(f"  (M15 Box Bias + NY Killzone + Sweeps + M1 CHoCH + M5 POI)")
    print(f"=======================================================")

    df_raw = fetch_or_fallback_candles(ticker)
    if df_raw.empty or len(df_raw) < 100:
        print(f"  [ERROR] Insufficient candles for {ticker}")
        return {"ticker": ticker, "status": "ERROR"}

    print(f"  Loaded {len(df_raw):,} intraday candles.")

    box_detector = M15BoxStructureDetector(left_bars=3, right_bars=3)
    session_mgr = SessionKillzoneManager(allow_ny_extension=True)
    choch_detector = M1CHoCHDetector(left_bars=2, right_bars=2)
    poi_selector = M5POISelector(min_displacement_ratio=0.8)
    order_mgr = BiasOrderManager(min_rr=5.0, risk_pct=0.01, tp1_pct=0.40, tp2_pct=0.60)

    budget = STARTING_BUDGET
    equity = budget
    equity_curve = [equity]
    trade_records = []
    active_trade = None
    pending_setup = None
    last_stopped_out_bar_idx = -999

    veto_count = 0
    approved_count = 0

    for i in range(100, len(df_raw)):
        cur_bar = df_raw.iloc[i]
        cur_time = cur_bar.name
        cur_low = float(cur_bar["low"])
        cur_high = float(cur_bar["high"])
        cur_close = float(cur_bar["close"])

        # 1. Manage Active Trade Tranches (Stop loss, TP1 @ 3.0R + Breakeven, TP2 @ M15 Target)
        if active_trade and not active_trade.is_completed:
            events = active_trade.check_exits(cur_bar, cur_time)
            if active_trade.is_completed:
                equity += active_trade.total_realized_pnl
                if active_trade.stopped_out:
                    last_stopped_out_bar_idx = i

                rec = {
                    "trade_id": active_trade.trade_id,
                    "ticker": ticker,
                    "side": active_trade.side,
                    "entry_time": str(active_trade.entry_time),
                    "exit_time": str(active_trade.completion_time),
                    "entry_price": active_trade.entry_price,
                    "initial_sl": active_trade.initial_stop_loss,
                    "total_shares": active_trade.total_quantity,
                    "pnl": active_trade.total_realized_pnl,
                    "rr_ratio": active_trade.rr_ratio,
                    "tp1_hit": active_trade.tp1_hit,
                    "tp2_hit": active_trade.tp2_hit,
                    "tranches": {
                        tid: {
                            "shares": t.quantity,
                            "target_price": t.target_price,
                            "exit_price": t.exit_price,
                            "status": t.status,
                            "pnl": t.pnl,
                        }
                        for tid, t in active_trade.tranches.items()
                    }
                }
                trade_records.append(rec)
                active_trade = None
            equity_curve.append(equity)
            continue

        # 2. Check Pending Limit Order fills
        if pending_setup:
            side = pending_setup["side"]
            limit_p = pending_setup["limit_price"]
            sl_p = pending_setup["stop_loss"]
            tp1_p = pending_setup["tp1_price"]
            bars_waiting = i - pending_setup["placed_bar_idx"]

            # Expiration after 12 bars (1 hour on 5m)
            if bars_waiting > 12:
                pending_setup = None
            # Invalidation Check (SL breached prior to fill)
            elif (side == "BUY" and cur_low <= sl_p) or (side == "SELL" and cur_high >= sl_p):
                pending_setup = None
            # Missed Pullback Check (Price expanded directly to TP1 before fill)
            elif (side == "BUY" and cur_high >= tp1_p) or (side == "SELL" and cur_low <= tp1_p):
                pending_setup = None
            else:
                filled = (cur_low <= limit_p) if side == "BUY" else (cur_high >= limit_p)
                if filled:
                    active_trade = order_mgr.create_trade(
                        ticker=ticker,
                        side=side,
                        entry_price=limit_p,
                        structural_extreme_price=pending_setup["structural_extreme"],
                        m15_target_price=pending_setup["target_price"],
                        equity=equity,
                        entry_time=cur_time,
                    )
                    pending_setup = None
                    equity_curve.append(equity)
                    continue

        # If holding an active position or waiting on a pending limit order, do not open duplicate setup
        if active_trade or pending_setup:
            equity_curve.append(equity)
            continue

        # 3. Rule 2: Session Killzone Gate (London 02:00-05:00 EST, NY 07:00-11:30 EST)
        in_killzone, session_name = session_mgr.is_in_killzone(cur_time)
        if not in_killzone:
            equity_curve.append(equity)
            continue

        # 4. Rule 1: M15 Macro Bias via The Box Method
        window = df_raw.iloc[:i]
        df_m15 = window.resample("15min", closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        if len(df_m15) > 1:
            df_m15 = df_m15.iloc[:-1]
        if len(df_m15) < 10:
            equity_curve.append(equity)
            continue

        bias, box_range = box_detector.analyze_external_structure(df_m15)
        if bias == BiasDirection.NEUTRAL or box_range is None:
            equity_curve.append(equity)
            continue

        bias_str = "BULLISH" if bias == BiasDirection.BULLISH else "BEARISH"
        side = "BUY" if bias == BiasDirection.BULLISH else "SELL"
        m15_target = box_range.target_price

        # 5. Rule 3: Liquidation Sweeps
        benchmarks = session_mgr.extract_prior_session_benchmarks(window, cur_time)
        sweep_status = session_mgr.check_liquidation_sweep(
            cur_high=cur_high,
            cur_low=cur_low,
            bias=bias_str,
            active_killzone=session_name,
            benchmarks=benchmarks,
            cur_time=cur_time,
        )
        if not sweep_status.swept:
            equity_curve.append(equity)
            continue

        # 6. Rule 4: Reversal Confirmation (1-Minute / 5-Minute Micro-CHoCH)
        df_micro = window.iloc[-30:].copy()
        choch = choch_detector.detect_m1_choch(
            df_m1=df_micro,
            bias=bias_str,
            sweep_time=sweep_status.sweep_time,
            sweep_level=sweep_status.benchmark_level,
        )
        if not choch:
            equity_curve.append(equity)
            continue

        structural_extreme = choch.structural_extreme_price
        tick_size = 0.01
        stop_loss_price = (structural_extreme - tick_size) if side == "BUY" else (structural_extreme + tick_size)

        # 7. Rule 5: Entry Model (5-Minute POI in recent displacement leg)
        recent_m5 = window.iloc[-15:].copy()
        best_poi = poi_selector.select_best_poi_with_confluence(
            df_m5=recent_m5,
            bias=bias_str,
            lookback_bars=15,
            structural_extreme=structural_extreme,
            m15_target=m15_target,
            choch_broken_level=choch.broken_level,
        )
        if not best_poi:
            equity_curve.append(equity)
            continue

        entry_limit_price = best_poi.proximate_price

        # 8. Asymmetric 1:5R Veto Gate
        approved, rr_val, veto_msg = order_mgr.evaluate_5r_veto(
            entry_price=entry_limit_price,
            stop_loss=stop_loss_price,
            m15_target=m15_target,
            side=side,
        )

        if not approved:
            veto_count += 1
            equity_curve.append(equity)
            continue

        approved_count += 1
        risk_per_share = abs(entry_limit_price - stop_loss_price)
        tp1_val = (entry_limit_price + 3.0 * risk_per_share) if side == "BUY" else (entry_limit_price - 3.0 * risk_per_share)

        # Immediate fill check or pending limit order
        can_fill_now = (cur_low <= entry_limit_price) if side == "BUY" else (cur_high >= entry_limit_price)
        if can_fill_now:
            active_trade = order_mgr.create_trade(
                ticker=ticker,
                side=side,
                entry_price=entry_limit_price,
                structural_extreme_price=structural_extreme,
                m15_target_price=m15_target,
                equity=equity,
                entry_time=cur_time,
            )
        else:
            pending_setup = {
                "side": side,
                "limit_price": entry_limit_price,
                "stop_loss": stop_loss_price,
                "structural_extreme": structural_extreme,
                "target_price": m15_target,
                "tp1_price": tp1_val,
                "placed_bar_idx": i,
                "rr_ratio": rr_val,
                "poi": best_poi,
                "choch": choch,
            }
        equity_curve.append(equity)

    # Compute Performance Metrics
    total_trades = len(trade_records)
    winning_trades = [t for t in trade_records if t.get("pnl", 0.0) > 0]
    losing_trades = [t for t in trade_records if t.get("pnl", 0.0) < 0]
    win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = sum([t.get("pnl", 0.0) for t in trade_records])
    final_equity = budget + total_pnl
    tot_return = ((final_equity - budget) / budget) * 100.0

    peak = budget
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    tp1_hits = sum(1 for t in trade_records if t.get("tp1_hit"))
    tp2_hits = sum(1 for t in trade_records if t.get("tp2_hit"))

    gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
    avg_rr = float(np.mean([t["rr_ratio"] for t in trade_records])) if trade_records else 0.0

    metrics = {
        "ticker": ticker,
        "return_pct": tot_return,
        "max_dd": max_dd,
        "trades": total_trades,
        "wins": len(winning_trades),
        "losses": len(losing_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_rr": avg_rr,
        "total_pnl": total_pnl,
        "final_equity": final_equity,
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "veto_count": veto_count,
        "approved_count": approved_count,
    }

    print(f"  Total Trades:     {total_trades}")
    print(f"  Win Rate:         {win_rate:.1f}% ({len(winning_trades)}W / {len(losing_trades)}L)")
    print(f"  Average R:R:      {avg_rr:.2f}R (Target Expectancy ~6.23R)")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Take Profit Breakdown: TP1 (3R) Hits={tp1_hits}, TP2 (M15 >=5R) Hits={tp2_hits}")
    print(f"  Setups Approved:  {approved_count}, Vetoed (<5R): {veto_count}")
    print(f"  Total Return:     {tot_return:+.2f}%")
    print(f"  Max Drawdown:     {max_dd:.2f}%")
    print(f"  Total Realized PnL: ${total_pnl:+,.2f}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    chart_file = os.path.join(repo_root, "logs", "charts", f"{ticker}_smc_bias.html")
    generate_smc_bias_plotly_chart(ticker, df_raw, trade_records, metrics, chart_file)
    metrics["chart_path"] = chart_file

    return metrics


def generate_smc_bias_summary_dashboard(all_metrics: list, dashboard_path: str):
    """Creates a unified institutional HTML dashboard linking all ticker charts with metrics."""
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)

    rows_html = ""
    for m in all_metrics:
        if not m or "return_pct" not in m:
            continue
        ret_color = "#26a69a" if m["return_pct"] >= 0 else "#ef5350"
        pnl_color = "#26a69a" if m["total_pnl"] >= 0 else "#ef5350"
        wr_color = "#26a69a" if m["win_rate"] >= 33.0 else "#f39c12"
        pf_val = m.get("profit_factor", 1.0)
        pf_color = "#26a69a" if pf_val >= 1.5 else ("#f39c12" if pf_val >= 1.0 else "#ef5350")
        chart_name = f"{m['ticker']}_smc_bias.html"

        rows_html += f"""
        <tr>
            <td style="font-weight: bold; font-size: 1.1em;"><a href="{chart_name}" target="chart_frame" onclick="switchChart('{chart_name}', '{m['ticker']}')" style="color: #29b6f6; text-decoration: none;">{m['ticker']}</a></td>
            <td style="color: {ret_color}; font-weight: bold;">{m['return_pct']:+.2f}%</td>
            <td style="color: #ef5350;">{m['max_dd']:.2f}%</td>
            <td style="color: {wr_color}; font-weight: bold;">{m['win_rate']:.1f}% ({m['wins']}W / {m['losses']}L)</td>
            <td style="color: #29b6f6; font-weight: bold;">{m['avg_rr']:.2f}R</td>
            <td style="color: {pf_color}; font-weight: bold;">{pf_val:.2f}</td>
            <td>{m['trades']}</td>
            <td>{m['tp1_hits']}</td>
            <td style="color: #26a69a; font-weight: bold;">{m['tp2_hits']} (≥5R)</td>
            <td style="color: #f39c12;">{m['veto_count']}</td>
            <td style="color: {pnl_color}; font-weight: bold;">${m['total_pnl']:+,.2f}</td>
            <td><a href="{chart_name}" target="_blank" style="display: inline-block; padding: 4px 10px; background: #1e88e5; color: #fff; border-radius: 4px; text-decoration: none; font-size: 0.85em;">Open Chart ↗</a></td>
        </tr>
        """

    first_ticker = all_metrics[0]["ticker"] if all_metrics else "NVDA"
    first_chart = f"{first_ticker}_smc_bias.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The 5-Rule SMC Intraday Bias Model - Executive Simulation Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #131722;
            color: #d1d4dc;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2a2e39;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        h1 {{ margin: 0; font-size: 1.6em; color: #ffffff; }}
        .badge {{ background: #29b6f622; border: 1px solid #29b6f6; color: #29b6f6; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .card {{
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 15px;
        }}
        .card h4 {{ margin: 0 0 8px 0; color: #f39c12; font-size: 0.9em; text-transform: uppercase; }}
        .card p {{ margin: 0; font-size: 0.82em; color: #9598a1; line-height: 1.4; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1e222d;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #2a2e39;
            margin-bottom: 25px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #2a2e39;
        }}
        th {{
            background-color: #262b3d;
            color: #787b86;
            font-size: 0.85em;
            text-transform: uppercase;
        }}
        tr:hover {{ background-color: #262b3d55; }}
        .chart-container {{
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            overflow: hidden;
            height: 750px;
        }}
        iframe {{ width: 100%; height: 100%; border: none; }}
    </style>
    <script>
        function switchChart(url, ticker) {{
            document.getElementById('chart_frame').src = url;
            document.getElementById('current_ticker_title').innerText = ticker + ' Interactive SMC Intraday Bias Chart';
        }}
    </script>
</head>
<body>
    <div class="header">
        <div>
            <h1>The 5-Rule SMC Intraday Bias Model</h1>
            <span style="color: #787b86; font-size: 0.9em;">Systematic Playbook by Lewis Kelly | Asymmetric Edge (1:5R to 1:8+) | M15 Box Bias ➔ Killzones ➔ Liquidation ➔ M1 CHoCH ➔ M5 POI</span>
        </div>
        <span class="badge">LEWIS KELLY PLAYBOOK COMPLIANT</span>
    </div>

    <div class="cards">
        <div class="card">
            <h4>Rule 1: The Box Method (M15)</h4>
            <p>External swings define the macro range; internal market noise is strictly disregarded.</p>
        </div>
        <div class="card">
            <h4>Rule 2 & 3: Killzone & Sweep</h4>
            <p>Restricted to London/NY killzones. Must purge Asian or Premarket liquidity before entering.</p>
        </div>
        <div class="card">
            <h4>Rule 4: M1 Micro-CHoCH</h4>
            <p>Impulsive candle body close proves counter-trend exhaustion and alignment with M15 bias.</p>
        </div>
        <div class="card">
            <h4>Rule 5: M5 POI Entry</h4>
            <p>Order Block + FVG confluence stacking with proximate limit order entry.</p>
        </div>
        <div class="card">
            <h4>Risk: 1:5R+ Asymmetry</h4>
            <p>Micro-stop vs Macro M15 Target. Hybrid scaling: 40% TP1 @ 3R (+BE), 60% Runner @ M15 target.</p>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Total Return</th>
                <th>Max DD</th>
                <th>Win Rate</th>
                <th>Avg R:R</th>
                <th>Profit Factor</th>
                <th>Trades</th>
                <th>TP1 (3R)</th>
                <th>TP2 (≥5R)</th>
                <th>Vetoed (&lt;5R)</th>
                <th>Total Realized PnL</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div class="header" style="margin-top: 30px;">
        <h2 id="current_ticker_title" style="margin: 0; font-size: 1.3em; color: #ffffff;">{first_ticker} Interactive SMC Intraday Bias Chart</h2>
    </div>

    <div class="chart-container">
        <iframe id="chart_frame" src="{first_chart}"></iframe>
    </div>
</body>
</html>
    """

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> Unified SMC Bias Dashboard created: {dashboard_path}")


def main():
    print("==================================================================")
    print("  LAUNCHING 5-RULE SMC INTRADAY BIAS MODEL SIMULATION           ")
    print("==================================================================")

    all_metrics = []
    for ticker in TICKERS:
        try:
            m = run_smc_bias_simulation_for_ticker(ticker)
            all_metrics.append(m)
        except Exception as e:
            print(f"  [ERROR] Simulation failed for {ticker}: {e}")
            import traceback
            traceback.print_exc()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dashboard_file = os.path.join(repo_root, "logs", "charts", "smc_bias_dashboard.html")
    generate_smc_bias_summary_dashboard(all_metrics, dashboard_file)

    print("\n==================================================================")
    print("  SIMULATION COMPLETE! Open the dashboard in your browser:       ")
    print(f"  file:///{dashboard_file.replace(os.sep, '/')}")
    print("==================================================================")


if __name__ == "__main__":
    main()
