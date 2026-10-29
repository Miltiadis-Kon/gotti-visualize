"""
Market Reactor Multi-Asset Backtest Runner
==========================================
Executes a multi-asset portfolio simulation of `market_reactor.py`
for all news signals received on a given date (default: yesterday, 2026-10-28).

Evaluates:
- Dual Hard Floor confidence filtering
- Selector assignment (1: 1H Macro Alignment, 2: 15m Catalyst Momentum, 3: 30m Directional Flow)
- S/R cluster calculation & ATR volatility optimization
- Realized execution starting from market open (9:30 AM EST)
- Intraday & overnight trajectory through today
"""

import os
import sys
import json
import logging
from math import floor
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReactorBacktest")


def run_portfolio_backtest(target_date: str = "2026-10-28", initial_portfolio: float = 100000.0, risk_pct: float = 0.02):
    print("=" * 85)
    print(f" MARKET REACTOR MULTI-ASSET PORTFOLIO BACKTEST — DATE: {target_date}")
    print(f" Portfolio: ${initial_portfolio:,.2f} | Risk per Trade: {risk_pct:.1%} (${initial_portfolio * risk_pct:,.2f})")
    print("=" * 85)

    # 1. Fetch & Classify Signals
    signals = fetch_reactor_signals(target_date)
    print(f"\n[STEP 1] Ingested {len(signals)} raw signals from MySQL for {target_date}.")

    eligible_signals = [s for s in signals if s.is_eligible]
    rejected_signals = [s for s in signals if not s.is_eligible]

    print(f"  -> Eligible Signals (>= 70% Dual Hard Floor): {len(eligible_signals)}")
    print(f"  -> Rejected Signals: {len(rejected_signals)}")
    for r in rejected_signals:
        print(f"     * {r.ticker}: Fund Conf = {r.fundamental_confidence:.1%}, News Conv = {r.news_conviction:.1%} (Below 70% threshold)")

    # 2. Iterate through each eligible signal and simulate trade execution
    print("\n[STEP 2] Simulating Market Open Execution & Trade Trajectories...")

    trade_results = []
    
    # We will simulate trading on market open of target_date (2026-10-28 09:30 EST)
    # Download 15m intraday data covering target_date and subsequent trading session
    for sig in eligible_signals:
        ticker = sig.ticker
        print(f"\nProcessing {ticker} | Side: {sig.signal_position} | Selector {sig.selector_id} ({sig.selector_name}, {sig.sr_timeframe})...")

        try:
            # Fetch intraday data from yfinance (15m interval)
            df_intraday = yf.download(ticker, period="5d", interval="15m", progress=False)
            if df_intraday.empty:
                print(f"  [WARN] No market data for {ticker}. Skipping.")
                continue

            # Flatten multi-index columns if present (from modern yfinance)
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = [c[0].lower() for c in df_intraday.columns]
            else:
                df_intraday.columns = [c.lower() for c in df_intraday.columns]

            # Localize / convert index to Eastern time
            if df_intraday.index.tz is not None:
                df_intraday.index = df_intraday.index.tz_convert("America/New_York")
            else:
                df_intraday.index = df_intraday.index.tz_localize("UTC").tz_convert("America/New_York")

            # Filter data starting from target_date market open (09:30 EST)
            target_dt = pd.to_datetime(target_date).date()
            sim_data = df_intraday[df_intraday.index.date >= target_dt]

            if sim_data.empty:
                print(f"  [WARN] No price action on/after {target_date} for {ticker}. Skipping.")
                continue

            # Get market open candle (first candle on or after 9:30 AM on target_date)
            open_candles = sim_data[sim_data.index.date == target_dt]
            if open_candles.empty:
                first_bar = sim_data.iloc[0]
            else:
                first_bar = open_candles.iloc[0]

            entry_time = first_bar.name
            entry_price = float(first_bar["open"])

            # 3. Calculate Key Levels for Selector resolution
            res_code = sig.resolution_code  # "1H" or "15m"
            try:
                kl_res = analyze(ticker, resolutions=[res_code], use_alpaca=False)
                kl_df = kl_res.key_levels
                
                supports = kl_df[(kl_df['type'] == 'support') & (kl_df['level_price'] < entry_price)]['level_price'].values
                resistances = kl_df[(kl_df['type'] == 'resistance') & (kl_df['level_price'] > entry_price)]['level_price'].values

                raw_sup = float(np.max(supports)) if len(supports) > 0 else entry_price * 0.98
                raw_res = float(np.min(resistances)) if len(resistances) > 0 else entry_price * 1.02
            except Exception as e:
                raw_sup = entry_price * 0.98
                raw_res = entry_price * 1.02

            # 4. Calculate ATR
            atr_val = calculate_atr(df_intraday, length=14)
            if atr_val is None or atr_val <= 0 or np.isnan(atr_val):
                atr_val = entry_price * 0.015

            # 5. Optimize SL / TP via StrategyBaseplate ATR framework
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

            # 6. Sizing
            risk_budget = initial_portfolio * risk_pct
            risk_per_share = abs(entry_price - sl_price)
            if risk_per_share <= 0:
                quantity = 1
            else:
                quantity = floor(risk_budget / risk_per_share)
            
            max_shares = floor((initial_portfolio * 0.95) / entry_price)
            quantity = max(1, min(quantity, max_shares))
            position_value = quantity * entry_price

            # 7. Walk forward through subsequent bars to evaluate trade outcome
            status = "OPEN (Holding Overnight)"
            exit_price = float(sim_data.iloc[-1]["close"])
            exit_time = sim_data.iloc[-1].name
            exit_reason = "ACTIVE_OVERNIGHT"

            # Check candle by candle from entry onwards
            for bar_idx in range(len(sim_data)):
                bar = sim_data.iloc[bar_idx]
                high = float(bar["high"])
                low = float(bar["low"])
                bar_time = bar.name

                if sig.signal_position == "BUY":
                    # Check Take Profit hit
                    if high >= tp_price:
                        status = "HIT TAKE PROFIT"
                        exit_price = tp_price
                        exit_time = bar_time
                        exit_reason = "TP"
                        break
                    # Check Stop Loss hit
                    if low <= sl_price:
                        status = "STOPPED OUT"
                        exit_price = sl_price
                        exit_time = bar_time
                        exit_reason = "SL"
                        break
                else: # SELL (Short)
                    # Check Take Profit hit
                    if low <= tp_price:
                        status = "HIT TAKE PROFIT"
                        exit_price = tp_price
                        exit_time = bar_time
                        exit_reason = "TP"
                        break
                    # Check Stop Loss hit
                    if high >= sl_price:
                        status = "STOPPED OUT"
                        exit_price = sl_price
                        exit_time = bar_time
                        exit_reason = "SL"
                        break

            # Calculate PnL
            if sig.signal_position == "BUY":
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) * quantity
                pnl_pct = (entry_price - exit_price) / entry_price

            record = {
                "ticker": ticker,
                "position": sig.signal_position,
                "selector_id": sig.selector_id,
                "selector_name": sig.selector_name,
                "timeframe": sig.sr_timeframe,
                "fund_conf": sig.fundamental_confidence,
                "news_conv": sig.news_conviction,
                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
                "entry_price": entry_price,
                "quantity": quantity,
                "position_val": position_value,
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "rr_ratio": rr_ratio,
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
                "exit_price": exit_price,
                "status": status,
                "exit_reason": exit_reason,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
            trade_results.append(record)

            print(f"  -> Filled {quantity} shares @ ${entry_price:.2f} | SL: ${sl_price:.2f} | TP: ${tp_price:.2f} (R:R {rr_ratio:.2f})")
            print(f"     Outcome: {status} @ ${exit_price:.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.2%})")

        except Exception as e:
            print(f"  [ERROR] Failed simulating {ticker}: {e}")

    # 3. Portfolio Summary & Output Table
    print("\n" + "=" * 85)
    print(" BACKTEST RESULTS SUMMARY")
    print("=" * 85)

    if not trade_results:
        print("No trade executions recorded.")
        return

    df_res = pd.DataFrame(trade_results)
    
    total_trades = len(df_res)
    tp_hits = len(df_res[df_res["status"] == "HIT TAKE PROFIT"])
    sl_hits = len(df_res[df_res["status"] == "STOPPED OUT"])
    open_trades = len(df_res[df_res["status"].str.contains("OPEN")])
    total_pnl = df_res["pnl"].sum()
    total_return_pct = total_pnl / initial_portfolio
    win_rate = (tp_hits / (tp_hits + sl_hits)) if (tp_hits + sl_hits) > 0 else 0.0

    # Display clean table
    display_cols = [
        "ticker", "position", "selector_id", "timeframe",
        "entry_price", "stop_loss", "take_profit", "exit_price",
        "status", "pnl", "pnl_pct"
    ]
    formatted_df = df_res[display_cols].copy()
    formatted_df["entry_price"] = formatted_df["entry_price"].map(lambda x: f"${x:.2f}")
    formatted_df["stop_loss"] = formatted_df["stop_loss"].map(lambda x: f"${x:.2f}")
    formatted_df["take_profit"] = formatted_df["take_profit"].map(lambda x: f"${x:.2f}")
    formatted_df["exit_price"] = formatted_df["exit_price"].map(lambda x: f"${x:.2f}")
    formatted_df["pnl"] = formatted_df["pnl"].map(lambda x: f"${x:+,.2f}")
    formatted_df["pnl_pct"] = formatted_df["pnl_pct"].map(lambda x: f"{x:+.2%}")

    print(formatted_df.to_string(index=False))

    print("\n" + "-" * 50)
    print(f" Total Trades Executed:    {total_trades}")
    print(f" Take Profit Targets Hit:  {tp_hits} ({tp_hits/total_trades:.1%})")
    print(f" Stop Losses Hit:          {sl_hits} ({sl_hits/total_trades:.1%})")
    print(f" Open (Overnight Runners): {open_trades} ({open_trades/total_trades:.1%})")
    print(f" Closed Win Rate:          {win_rate:.1%}")
    print(f" Net Portfolio P&L:        ${total_pnl:+,.2f}")
    print(f" Total Portfolio Return:   {total_return_pct:+.2%}")
    print("-" * 50)

    # Save to logs
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    out_csv = os.path.join(BASE_DIR, "logs", f"market_reactor_backtest_{target_date}.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[OK] Detailed backtest trade log saved to: {out_csv}")

    return df_res


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "2026-10-28"
    run_portfolio_backtest(target_date=target)
