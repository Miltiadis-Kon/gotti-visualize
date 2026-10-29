"""
Market Reactor Multi-Asset Backtest Runner
==========================================
Executes a multi-asset portfolio simulation of `market_reactor.py`
for news signals received across a specified date range or trading week.

Features:
- Dual Hard Floor confidence filtering (Fundamental >= 70% AND News Conviction >= 70%)
- Selector assignment (1: 1H Macro Alignment, 2: 15m Catalyst Momentum, 3: 30m Directional Flow)
- Dynamic S/R clustering via Key Levels engine + ATR volatility optimization
- True multi-day position carrying with overnight holding
- Separate calculation of:
    * Total Realized / Closed P&L
    * Total Unrealized / Open P&L
    * Total Net Portfolio P&L
"""

import os
import sys
import json
import logging
from math import floor
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
import yfinance as yf

# Set project root in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategies.market_reactor import (
    fetch_reactor_signals,
    MarketReactorClassifier,
    ReactorSignal
)
from strategies.key_levels import analyze
from strategies.risk_management import optimize_sl_tp, calculate_atr

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReactorBacktest")


def run_multi_day_backtest(
    date_list: List[str],
    initial_portfolio: float = 100000.0,
    risk_pct: float = 0.02,
    max_signals_per_day: Optional[int] = 15
) -> Dict[str, Any]:
    """
    Simulates a multi-day portfolio backtest over a list of trading dates.
    Carries open positions overnight until TP or SL is triggered.
    """
    sorted_dates = sorted(date_list)
    print("=" * 90)
    print(f" MARKET REACTOR MULTI-DAY / WEEKLY PORTFOLIO BACKTEST")
    print(f" Date Range: {sorted_dates[0]} to {sorted_dates[-1]} ({len(sorted_dates)} trading sessions)")
    print(f" Initial Portfolio: ${initial_portfolio:,.2f} | Risk per Trade: {risk_pct:.1%} (${initial_portfolio * risk_pct:,.2f})")
    print("=" * 90)

    # 1. Ingest & Classify signals for all requested dates
    all_eligible_signals: List[Tuple[str, ReactorSignal]] = []
    total_raw_signals = 0
    total_rejected = 0

    for d_str in sorted_dates:
        sigs = fetch_reactor_signals(d_str)
        total_raw_signals += len(sigs)
        eligible = [s for s in sigs if s.is_eligible]
        rejected = [s for s in sigs if not s.is_eligible]
        total_rejected += len(rejected)

        # Sort eligible by composite score (fund_conf * news_conv) descending
        eligible.sort(key=lambda x: (x.fundamental_confidence * x.news_conviction), reverse=True)
        if max_signals_per_day and len(eligible) > max_signals_per_day:
            eligible = eligible[:max_signals_per_day]

        for s in eligible:
            all_eligible_signals.append((d_str, s))

    print(f"\n[STEP 1] Ingested {total_raw_signals} total signals across {len(sorted_dates)} dates.")
    print(f"  -> Total Eligible Setups (>= 70% Dual Floor): {len(all_eligible_signals)}")
    print(f"  -> Total Filtered Out: {total_rejected}")

    if not all_eligible_signals:
        print("[WARN] No eligible signals found for the specified date range.")
        return {}

    # 2. Collect unique tickers to fetch price data in bulk
    unique_tickers = list(dict.fromkeys([s.ticker for _, s in all_eligible_signals]))
    print(f"  -> Unique Tickers to Evaluate: {len(unique_tickers)}")

    # Determine date bounds for historical data
    start_dt = pd.to_datetime(sorted_dates[0]) - timedelta(days=5)
    end_dt = pd.to_datetime(sorted_dates[-1]) + timedelta(days=2)

    print(f"\n[STEP 2] Fetching historical price action from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")

    # Fetch 15m market data
    ticker_data_map: Dict[str, pd.DataFrame] = {}
    for ticker in unique_tickers:
        try:
            df = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="15m", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

                if df.index.tz is not None:
                    df.index = df.index.tz_convert("America/New_York")
                else:
                    df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

                ticker_data_map[ticker] = df
        except Exception:
            pass

    print(f"  -> Successfully loaded intraday market data for {len(ticker_data_map)} / {len(unique_tickers)} tickers.")

    # 3. Simulate trade execution day-by-day
    print("\n[STEP 3] Simulating Market Open Execution & Multi-Day Overnight Trajectories...")

    active_positions: Dict[str, Dict[str, Any]] = {}
    completed_trades: List[Dict[str, Any]] = []

    for d_str in sorted_dates:
        curr_date = pd.to_datetime(d_str).date()
        daily_sigs = [s for d, s in all_eligible_signals if d == d_str]

        # A. Open new positions for signals of day d_str at market open (9:30 AM EST)
        for sig in daily_sigs:
            ticker = sig.ticker
            if ticker in active_positions:
                continue  # Already holding position from prior day
            if ticker not in ticker_data_map:
                continue

            df = ticker_data_map[ticker]
            day_candles = df[df.index.date == curr_date]
            if day_candles.empty:
                continue

            first_bar = day_candles.iloc[0]
            entry_time = first_bar.name
            entry_price = float(first_bar["open"])

            # Compute S/R levels
            try:
                kl_res = analyze(ticker, resolutions=[sig.resolution_code], use_alpaca=False)
                kl_df = kl_res.key_levels
                supports = kl_df[(kl_df['type'] == 'support') & (kl_df['level_price'] < entry_price)]['level_price'].values
                resistances = kl_df[(kl_df['type'] == 'resistance') & (kl_df['level_price'] > entry_price)]['level_price'].values

                raw_sup = float(np.max(supports)) if len(supports) > 0 else entry_price * 0.98
                raw_res = float(np.min(resistances)) if len(resistances) > 0 else entry_price * 1.02
            except Exception:
                raw_sup = entry_price * 0.98
                raw_res = entry_price * 1.02

            # Compute ATR
            atr_val = calculate_atr(df[df.index <= entry_time], length=14)
            if atr_val is None or atr_val <= 0 or np.isnan(atr_val):
                atr_val = entry_price * 0.015

            # Optimize SL / TP
            side = "buy" if sig.signal_position == "BUY" else "sell"
            raw_sl = raw_sup if sig.signal_position == "BUY" else raw_res
            raw_tp = raw_res if sig.signal_position == "BUY" else raw_sup

            risk_levels = optimize_sl_tp(
                entry_price=entry_price,
                stop_loss=raw_sl,
                take_profit=raw_tp,
                side=side,
                trading_style="swing_trading",
                atr=atr_val,
                risk_reward_ratio=1.5
            )

            sl_price = float(risk_levels.stop_loss)
            tp_price = float(risk_levels.take_profit)
            rr_ratio = float(risk_levels.risk_reward_ratio)

            # Sizing (2% risk)
            risk_budget = initial_portfolio * risk_pct
            risk_per_share = abs(entry_price - sl_price)
            qty = floor(risk_budget / max(0.01, risk_per_share))
            max_shares = floor((initial_portfolio * 0.95) / entry_price)
            quantity = max(1, min(qty, max_shares))

            active_positions[ticker] = {
                "ticker": ticker,
                "position": sig.signal_position,
                "selector_id": sig.selector_id,
                "selector_name": sig.selector_name,
                "timeframe": sig.sr_timeframe,
                "fund_conf": sig.fundamental_confidence,
                "news_conv": sig.news_conviction,
                "entry_time": entry_time,
                "entry_date": d_str,
                "entry_price": entry_price,
                "quantity": quantity,
                "position_val": quantity * entry_price,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "rr_ratio": rr_ratio,
                "current_price": entry_price,
            }

        # B. Walk forward intraday candles for all active positions on day curr_date
        closed_tickers = []
        for ticker, pos in active_positions.items():
            if ticker not in ticker_data_map:
                continue
            df = ticker_data_map[ticker]
            # Candles for today after or at entry
            eval_candles = df[(df.index.date == curr_date) & (df.index >= pos["entry_time"])]

            for bar_idx in range(len(eval_candles)):
                bar = eval_candles.iloc[bar_idx]
                high = float(bar["high"])
                low = float(bar["low"])
                bar_time = bar.name

                pos["current_price"] = float(bar["close"])

                # Check BUY exit conditions
                if pos["position"] == "BUY":
                    if high >= pos["take_profit"]:
                        pos["status"] = "HIT TAKE PROFIT"
                        pos["exit_price"] = pos["take_profit"]
                        pos["exit_time"] = bar_time
                        pos["exit_reason"] = "TP"
                        pos["pnl"] = (pos["take_profit"] - pos["entry_price"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["take_profit"] - pos["entry_price"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                    elif low <= pos["stop_loss"]:
                        pos["status"] = "STOPPED OUT"
                        pos["exit_price"] = pos["stop_loss"]
                        pos["exit_time"] = bar_time
                        pos["exit_reason"] = "SL"
                        pos["pnl"] = (pos["stop_loss"] - pos["entry_price"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["stop_loss"] - pos["entry_price"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                # Check SELL exit conditions
                else:
                    if low <= pos["take_profit"]:
                        pos["status"] = "HIT TAKE PROFIT"
                        pos["exit_price"] = pos["take_profit"]
                        pos["exit_time"] = bar_time
                        pos["exit_reason"] = "TP"
                        pos["pnl"] = (pos["entry_price"] - pos["take_profit"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["entry_price"] - pos["take_profit"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                    elif high >= pos["stop_loss"]:
                        pos["status"] = "STOPPED OUT"
                        pos["exit_price"] = pos["stop_loss"]
                        pos["exit_time"] = bar_time
                        pos["exit_reason"] = "SL"
                        pos["pnl"] = (pos["entry_price"] - pos["stop_loss"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["entry_price"] - pos["stop_loss"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break

        # Remove closed positions
        for t in closed_tickers:
            del active_positions[t]

    # 4. Final Accounting: Closed vs Open (Overnight) Positions
    open_trades: List[Dict[str, Any]] = []
    for ticker, pos in active_positions.items():
        if ticker in ticker_data_map and not ticker_data_map[ticker].empty:
            latest_price = float(ticker_data_map[ticker].iloc[-1]["close"])
            latest_time = ticker_data_map[ticker].iloc[-1].name
        else:
            latest_price = pos["current_price"]
            latest_time = pos["entry_time"]

        if pos["position"] == "BUY":
            unrealized_pnl = (latest_price - pos["entry_price"]) * pos["quantity"]
            unrealized_pct = (latest_price - pos["entry_price"]) / pos["entry_price"]
        else:
            unrealized_pnl = (pos["entry_price"] - latest_price) * pos["quantity"]
            unrealized_pct = (pos["entry_price"] - latest_price) / pos["entry_price"]

        pos["status"] = "OPEN (Holding Overnight)"
        pos["exit_price"] = latest_price
        pos["exit_time"] = latest_time
        pos["exit_reason"] = "ACTIVE_OVERNIGHT"
        pos["pnl"] = unrealized_pnl
        pos["pnl_pct"] = unrealized_pct
        open_trades.append(pos)

    # 5. Calculate Metrics
    df_closed = pd.DataFrame(completed_trades)
    df_open = pd.DataFrame(open_trades)

    closed_pnl = float(df_closed["pnl"].sum()) if not df_closed.empty else 0.0
    open_pnl = float(df_open["pnl"].sum()) if not df_open.empty else 0.0
    total_net_pnl = closed_pnl + open_pnl
    total_return_pct = total_net_pnl / initial_portfolio

    total_closed_count = len(df_closed)
    tp_count = len(df_closed[df_closed["status"] == "HIT TAKE PROFIT"]) if not df_closed.empty else 0
    sl_count = len(df_closed[df_closed["status"] == "STOPPED OUT"]) if not df_closed.empty else 0
    win_rate = (tp_count / total_closed_count) if total_closed_count > 0 else 0.0

    gross_profits = float(df_closed[df_closed["pnl"] > 0]["pnl"].sum()) if not df_closed.empty else 0.0
    gross_losses = float(abs(df_closed[df_closed["pnl"] < 0]["pnl"].sum())) if not df_closed.empty else 0.0
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (999.0 if gross_profits > 0 else 0.0)

    # 6. Print Executive Summary
    print("\n" + "=" * 90)
    print(" EXECUTIVE PORTFOLIO SUMMARY — MARKET REACTOR")
    print("=" * 90)

    print(f"\n{'METRIC':<35} {'VALUE':<25}")
    print("-" * 60)
    print(f"{'Total Signals Executed':<35} {len(completed_trades) + len(open_trades)}")
    print(f"{'Closed Trades':<35} {total_closed_count}")
    print(f"{'  -> Take Profit Targets Hit':<35} {tp_count} ({win_rate:.1%})")
    print(f"{'  -> Stopped Out Trades':<35} {sl_count} ({sl_count/max(1, total_closed_count):.1%})")
    print(f"{'Active Open Positions':<35} {len(open_trades)}")
    print(f"{'Profit Factor':<35} {profit_factor:.2f}")
    print("-" * 60)
    print(f"{'TOTAL CLOSED P&L (Realized)':<35} ${closed_pnl:+,.2f}")
    print(f"{'TOTAL OPEN P&L (Unrealized)':<35} ${open_pnl:+,.2f}")
    print(f"{'TOTAL NET P&L (Combined)':<35} ${total_net_pnl:+,.2f} ({total_return_pct:+.2%})")
    print("-" * 60)

    # Print Open Positions Detail
    if not df_open.empty:
        print("\n ACTIVE OPEN POSITIONS (Holding Overnight):")
        open_display = df_open[["ticker", "position", "entry_date", "entry_price", "current_price", "stop_loss", "take_profit", "pnl", "pnl_pct"]].copy()
        open_display["entry_price"] = open_display["entry_price"].map(lambda x: f"${x:.2f}")
        open_display["current_price"] = open_display["current_price"].map(lambda x: f"${x:.2f}")
        open_display["stop_loss"] = open_display["stop_loss"].map(lambda x: f"${x:.2f}")
        open_display["take_profit"] = open_display["take_profit"].map(lambda x: f"${x:.2f}")
        open_display["pnl"] = open_display["pnl"].map(lambda x: f"${x:+,.2f}")
        open_display["pnl_pct"] = open_display["pnl_pct"].map(lambda x: f"{x:+.2%}")
        print(open_display.to_string(index=False))

    # Print Closed Trades Detail
    if not df_closed.empty:
        print("\n CLOSED TRADES BREAKDOWN:")
        closed_display = df_closed[["ticker", "position", "entry_date", "entry_price", "exit_price", "status", "pnl", "pnl_pct"]].copy()
        closed_display["entry_price"] = closed_display["entry_price"].map(lambda x: f"${x:.2f}")
        closed_display["exit_price"] = closed_display["exit_price"].map(lambda x: f"${x:.2f}")
        closed_display["pnl"] = closed_display["pnl"].map(lambda x: f"${x:+,.2f}")
        closed_display["pnl_pct"] = closed_display["pnl_pct"].map(lambda x: f"{x:+.2%}")
        print(closed_display.to_string(index=False))

    # Save consolidated records
    all_records = completed_trades + open_trades
    if all_records:
        df_all = pd.DataFrame(all_records)
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        out_csv = os.path.join(BASE_DIR, "logs", f"market_reactor_weekly_{sorted_dates[0]}_to_{sorted_dates[-1]}.csv")
        df_all.to_csv(out_csv, index=False)
        print(f"\n[OK] Full trade ledger saved to: {out_csv}")

    return {
        "closed_pnl": closed_pnl,
        "open_pnl": open_pnl,
        "total_pnl": total_net_pnl,
        "total_return_pct": total_return_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": len(all_records),
        "closed_trades": total_closed_count,
        "open_trades": len(open_trades),
        "df_closed": df_closed,
        "df_open": df_open
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "recent_week"

    if mode == "oct_week":
        # Full 5-day continuous trading week (Oct 14 - Oct 20, 2026)
        trading_dates = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"]
        run_multi_day_backtest(trading_dates, max_signals_per_day=8)
    else:
        # Recent calendar week (Oct 22 - Oct 29, 2026), testing active signals
        trading_dates = ["2026-10-28", "2026-10-29"]
        run_multi_day_backtest(trading_dates, max_signals_per_day=15)
