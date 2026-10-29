"""
Combined News Strategies Backtest Runner (run_combined_news_backtest.py)
========================================================================
Runs comparative multi-day portfolio backtests for:
1. Market Reactor (Baseline - Open Execution + Selector S/R)
2. News + MultiTimeframe Key Levels & Fibonacci
3. News + Liquidity Sweep Engine

Evaluates both:
- Full 5-Trading-Day Week (Oct 14 - Oct 20, 2026)
- Recent Calendar Week (Oct 28 - Oct 29, 2026)

Reports:
- Total Closed P&L (Realized)
- Total Open P&L (Unrealized Overnight)
- Total Combined Net P&L
- Win Rate, Profit Factor, Trade Breakdown
"""

import os
import sys
import json
import logging
from math import floor
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategies.market_reactor import fetch_reactor_signals, ReactorSignal
from strategies.key_levels.key_levels import find_key_levels, merge_key_levels
from strategies.key_levels.fibonacci_levels import get_fibonacci_trade_setups
from strategies.risk_management import optimize_sl_tp, calculate_atr

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


def simulate_strategy_engine(
    strategy_name: str,
    date_list: List[str],
    all_eligible_signals: List[Tuple[str, ReactorSignal]],
    ticker_data_map: Dict[str, pd.DataFrame],
    all_market_dates: Optional[List[str]] = None,
    initial_portfolio: float = 100000.0,
    risk_pct: float = 0.02
) -> Dict[str, Any]:
    """
    Simulates a specific technical engine gated by news direction over multiple days.
    Accurately carries active positions overnight across all market trading sessions.
    """
    sorted_dates = sorted(all_market_dates) if all_market_dates else sorted(date_list)
    active_positions: Dict[str, Dict[str, Any]] = {}
    completed_trades: List[Dict[str, Any]] = []

    for d_str in sorted_dates:
        curr_date = pd.to_datetime(d_str).date()
        daily_sigs = [s for d, s in all_eligible_signals if d == d_str]

        for sig in daily_sigs:
            ticker = sig.ticker
            if ticker in active_positions or ticker not in ticker_data_map:
                continue

            df = ticker_data_map[ticker]
            day_candles = df[df.index.date == curr_date]
            if day_candles.empty:
                continue

            allowed_side = sig.signal_position  # "BUY" or "SELL"
            first_bar = day_candles.iloc[0]
            open_price = float(first_bar["open"])
            open_time = first_bar.name

            # ── 1. ENGINE 1: MARKET REACTOR BASELINE ─────────────────────────
            if strategy_name == "MarketReactor":
                entry_price = open_price
                entry_time = open_time

                # Determine S/R & ATR based on selector timeframe
                atr_val = calculate_atr(df[df.index <= entry_time], length=14) or (entry_price * 0.015)
                tf = getattr(sig, "sr_timeframe", "1H")
                prior_bars = df[df.index < entry_time]
                if tf == "15m":
                    recent_bars = prior_bars.tail(16)
                elif tf == "30m":
                    recent_bars = prior_bars.tail(28)
                else:  # 1H
                    recent_bars = prior_bars.tail(40)

                sup = float(recent_bars["low"].min()) if not recent_bars.empty else entry_price * 0.98
                res = float(recent_bars["high"].max()) if not recent_bars.empty else entry_price * 1.02

                side = "buy" if allowed_side == "BUY" else "sell"
                raw_sl = sup if allowed_side == "BUY" else res
                raw_tp = res if allowed_side == "BUY" else sup

                rl = optimize_sl_tp(
                    entry_price=entry_price,
                    stop_loss=raw_sl,
                    take_profit=raw_tp,
                    side=side,
                    atr=atr_val,
                    trading_style="swing_trading",
                    risk_reward_ratio=1.5
                )
                sl_price, tp_price = float(rl.stop_loss), float(rl.take_profit)

                risk_per_share = abs(entry_price - sl_price)
                qty = floor((initial_portfolio * risk_pct) / max(0.01, risk_per_share))
                qty = max(1, min(qty, floor((initial_portfolio * 0.95) / entry_price)))

                active_positions[ticker] = {
                    "ticker": ticker, "position": allowed_side, "entry_date": d_str,
                    "entry_time": entry_time, "entry_price": entry_price, "quantity": qty,
                    "stop_loss": sl_price, "take_profit": tp_price, "current_price": entry_price
                }

            # ── 2. ENGINE 2: NEWS + MULTI-TIMEFRAME KEY LEVELS & FIBONACCI ──
            elif strategy_name == "News_MTF_KeyLevels":
                # Compute multi-TF levels up to market open
                prior_df = df[df.index < open_time]
                if len(prior_df) < 20:
                    continue

                atr_val = calculate_atr(prior_df, length=14) or (open_price * 0.015)
                atr_buf = atr_val * 0.5

                # Generate 1D, 15m levels
                df_1d = prior_df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
                df_15m = prior_df.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

                found_levels = []
                for tf_code, r_df in [("1D", df_1d), ("15m", df_15m)]:
                    if len(r_df) >= 10:
                        l = find_key_levels(r_df, resolution=tf_code)
                        if not l.empty:
                            found_levels.append(l)

                merged_l = merge_key_levels(pd.concat(found_levels, ignore_index=True)) if found_levels else pd.DataFrame()
                sups = merged_l[merged_l["type"] == "support"].sort_values("level_price", ascending=False) if not merged_l.empty else pd.DataFrame()
                resists = merged_l[merged_l["type"] == "resistance"].sort_values("level_price", ascending=True) if not merged_l.empty else pd.DataFrame()

                # Fibonacci setups
                fib_setups = get_fibonacci_trade_setups(df_15m, resolution="15m", backcandles=30) if len(df_15m) >= 15 else pd.DataFrame()

                # Scan RTH candles on curr_date for trigger
                for idx in range(len(day_candles)):
                    bar = day_candles.iloc[idx]
                    p = float(bar["close"])
                    t = bar.name

                    # Check Fib setup in allowed side
                    matched = False
                    if not fib_setups.empty:
                        for _, f_row in fib_setups.iterrows():
                            f_side = "BUY" if f_row["trend"] == "uptrend" else "SELL"
                            if f_side == allowed_side and abs(p - float(f_row["entry_price"])) <= (float(f_row["entry_price"]) * 0.006):
                                raw_sl = float(f_row["stop_loss"])
                                raw_tp = float(f_row["take_profit"])
                                sl_price = raw_sl - atr_buf if f_side == "BUY" else raw_sl + atr_buf
                                tp_price = raw_tp
                                matched = True
                                break

                    # Check Key Level bounce
                    if not matched:
                        if allowed_side == "BUY" and not sups.empty:
                            nearest_s = float(sups.iloc[0]["level_price"])
                            if abs(p - nearest_s) <= (nearest_s * 0.006):
                                sl_price = nearest_s - atr_buf
                                tp_price = float(resists.iloc[0]["level_price"]) if not resists.empty else p + (atr_val * 2.0)
                                matched = True
                        elif allowed_side == "SELL" and not resists.empty:
                            nearest_r = float(resists.iloc[0]["level_price"])
                            if abs(p - nearest_r) <= (nearest_r * 0.006):
                                sl_price = nearest_r + atr_buf
                                tp_price = float(sups.iloc[0]["level_price"]) if not sups.empty else p - (atr_val * 2.0)
                                matched = True

                    if matched and abs(tp_price - p) >= (1.5 * abs(p - sl_price)):
                        risk_per_share = abs(p - sl_price)
                        qty = floor((initial_portfolio * risk_pct) / max(0.01, risk_per_share))
                        qty = max(1, min(qty, floor((initial_portfolio * 0.95) / p)))

                        active_positions[ticker] = {
                            "ticker": ticker, "position": allowed_side, "entry_date": d_str,
                            "entry_time": t, "entry_price": p, "quantity": qty,
                            "stop_loss": sl_price, "take_profit": tp_price, "current_price": p
                        }
                        break

            # ── 3. ENGINE 3: NEWS + LIQUIDITY SWEEP ──────────────────────────
            elif strategy_name == "News_LiquiditySweep":
                prior_dates = sorted(list(set(df[df.index.date < curr_date].index.date)))
                if not prior_dates:
                    continue

                last_d = prior_dates[-1]
                prev_day_bars = df[df.index.date == last_d]
                if prev_day_bars.empty:
                    continue

                pdh = float(prev_day_bars["high"].max())
                pdl = float(prev_day_bars["low"].min())
                atr_val = calculate_atr(df[df.index <= open_time], length=14) or (open_price * 0.015)
                atr_buf = atr_val * 0.5

                vol_ma = day_candles["volume"].rolling(20).mean()

                # Scan RTH candles for sweep & reclaim
                swept = False
                sweep_ext = None
                sweep_bars = 0

                for idx in range(len(day_candles)):
                    bar = day_candles.iloc[idx]
                    h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
                    v = float(bar["volume"])
                    v_spike = v >= vol_ma.iloc[idx] if not np.isnan(vol_ma.iloc[idx]) else True
                    t = bar.name

                    if allowed_side == "BUY":
                        if not swept and l < pdl and v_spike:
                            swept = True
                            sweep_ext = l
                            sweep_bars = 1
                            continue
                        elif swept:
                            sweep_bars += 1
                            sweep_ext = min(sweep_ext, l)
                            # Reclaim inside PDL
                            if c > pdl:
                                sl_price = sweep_ext - atr_buf
                                tp_price = pdh
                                if (tp_price - c) >= (1.5 * (c - sl_price)):
                                    risk_per_share = abs(c - sl_price)
                                    qty = floor((initial_portfolio * risk_pct) / max(0.01, risk_per_share))
                                    qty = max(1, min(qty, floor((initial_portfolio * 0.95) / c)))

                                    active_positions[ticker] = {
                                        "ticker": ticker, "position": "BUY", "entry_date": d_str,
                                        "entry_time": t, "entry_price": c, "quantity": qty,
                                        "stop_loss": sl_price, "take_profit": tp_price, "current_price": c
                                    }
                                    break
                            if sweep_bars > 4:
                                swept = False

                    elif allowed_side == "SELL":
                        if not swept and h > pdh and v_spike:
                            swept = True
                            sweep_ext = h
                            sweep_bars = 1
                            continue
                        elif swept:
                            sweep_bars += 1
                            sweep_ext = max(sweep_ext, h)
                            # Reclaim back below PDH
                            if c < pdh:
                                sl_price = sweep_ext + atr_buf
                                tp_price = pdl
                                if (c - tp_price) >= (1.5 * (sl_price - c)):
                                    risk_per_share = abs(sl_price - c)
                                    qty = floor((initial_portfolio * risk_pct) / max(0.01, risk_per_share))
                                    qty = max(1, min(qty, floor((initial_portfolio * 0.95) / c)))

                                    active_positions[ticker] = {
                                        "ticker": ticker, "position": "SELL", "entry_date": d_str,
                                        "entry_time": t, "entry_price": c, "quantity": qty,
                                        "stop_loss": sl_price, "take_profit": tp_price, "current_price": c
                                    }
                                    break
                            if sweep_bars > 4:
                                swept = False

        # Evaluate intraday price action against TP/SL for all active positions
        closed_tickers = []
        for ticker, pos in active_positions.items():
            if ticker not in ticker_data_map:
                continue
            eval_candles = ticker_data_map[ticker][(ticker_data_map[ticker].index.date == curr_date) & (ticker_data_map[ticker].index >= pos["entry_time"])]

            for b_idx in range(len(eval_candles)):
                bar = eval_candles.iloc[b_idx]
                h, l = float(bar["high"]), float(bar["low"])
                pos["current_price"] = float(bar["close"])

                if pos["position"] == "BUY":
                    if h >= pos["take_profit"]:
                        pos["status"] = "HIT TAKE PROFIT"
                        pos["exit_price"] = pos["take_profit"]
                        pos["exit_time"] = bar.name
                        pos["pnl"] = (pos["take_profit"] - pos["entry_price"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["take_profit"] - pos["entry_price"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                    elif l <= pos["stop_loss"]:
                        pos["status"] = "STOPPED OUT"
                        pos["exit_price"] = pos["stop_loss"]
                        pos["exit_time"] = bar.name
                        pos["pnl"] = (pos["stop_loss"] - pos["entry_price"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["stop_loss"] - pos["entry_price"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                else: # SELL
                    if l <= pos["take_profit"]:
                        pos["status"] = "HIT TAKE PROFIT"
                        pos["exit_price"] = pos["take_profit"]
                        pos["exit_time"] = bar.name
                        pos["pnl"] = (pos["entry_price"] - pos["take_profit"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["entry_price"] - pos["take_profit"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                    elif h >= pos["stop_loss"]:
                        pos["status"] = "STOPPED OUT"
                        pos["exit_price"] = pos["stop_loss"]
                        pos["exit_time"] = bar.name
                        pos["pnl"] = (pos["entry_price"] - pos["stop_loss"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["entry_price"] - pos["stop_loss"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break

        for t in closed_tickers:
            del active_positions[t]

    # Evaluate remaining open positions at latest available price
    open_trades: List[Dict[str, Any]] = []
    for ticker, pos in active_positions.items():
        if ticker in ticker_data_map and not ticker_data_map[ticker].empty:
            last_p = float(ticker_data_map[ticker].iloc[-1]["close"])
            last_t = ticker_data_map[ticker].iloc[-1].name
        else:
            last_p = pos["current_price"]
            last_t = pos["entry_time"]

        if pos["position"] == "BUY":
            u_pnl = (last_p - pos["entry_price"]) * pos["quantity"]
            u_pct = (last_p - pos["entry_price"]) / pos["entry_price"]
        else:
            u_pnl = (pos["entry_price"] - last_p) * pos["quantity"]
            u_pct = (pos["entry_price"] - last_p) / pos["entry_price"]

        pos["status"] = "OPEN (Holding Overnight)"
        pos["exit_price"] = last_p
        pos["exit_time"] = last_t
        pos["pnl"] = u_pnl
        pos["pnl_pct"] = u_pct
        open_trades.append(pos)

    # Accounting
    df_c = pd.DataFrame(completed_trades)
    df_o = pd.DataFrame(open_trades)

    c_pnl = float(df_c["pnl"].sum()) if not df_c.empty else 0.0
    o_pnl = float(df_o["pnl"].sum()) if not df_o.empty else 0.0
    net_pnl = c_pnl + o_pnl
    tot_trades = len(completed_trades) + len(open_trades)
    tp_hits = len(df_c[df_c["status"] == "HIT TAKE PROFIT"]) if not df_c.empty else 0
    sl_hits = len(df_c[df_c["status"] == "STOPPED OUT"]) if not df_c.empty else 0
    win_rate = (tp_hits / len(df_c)) if len(df_c) > 0 else 0.0

    gross_profit = float(df_c[df_c["pnl"] > 0]["pnl"].sum()) if not df_c.empty else 0.0
    gross_loss = float(abs(df_c[df_c["pnl"] < 0]["pnl"].sum())) if not df_c.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "strategy": strategy_name,
        "total_trades": tot_trades,
        "closed_trades": len(df_c),
        "open_trades": len(open_trades),
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "win_rate": win_rate,
        "closed_pnl": c_pnl,
        "open_pnl": o_pnl,
        "net_pnl": net_pnl,
        "return_pct": net_pnl / initial_portfolio,
        "profit_factor": profit_factor,
        "df_closed": df_c,
        "df_open": df_o
    }


def run_comparative_backtest(period_mode: str = "month"):
    periods = []
    if period_mode in ("month", "1month", "all"):
        month_dates = [
            "2026-09-29", "2026-09-30", "2026-10-01",
            "2026-10-14", "2026-10-15", "2026-10-16",
            "2026-10-19", "2026-10-20", "2026-10-28", "2026-10-29"
        ]
        periods.append(("Full 1-Month Period (Sep 29 - Oct 29, 2026)", month_dates, 8))
    if period_mode in ("oct_week", "both", "all"):
        periods.append(("Full 5-Day Trading Week", ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"], 10))
    if period_mode in ("recent_week", "both", "all"):
        periods.append(("Recent Calendar Week", ["2026-10-28", "2026-10-29"], 15))

    initial_portfolio = 100000.0

    for period_title, date_list, max_per_day in periods:
        print("\n" + "=" * 105)
        print(f" EXECUTING COMPARATIVE EVALUATION: {period_title} ({date_list[0]} to {date_list[-1]})")
        print("=" * 105)

        # Ingest eligible signals
        all_eligible: List[Tuple[str, ReactorSignal]] = []
        for d in date_list:
            sigs = fetch_reactor_signals(d)
            elig = [s for s in sigs if s.is_eligible]
            elig.sort(key=lambda x: (x.fundamental_confidence * x.news_conviction), reverse=True)
            if max_per_day and len(elig) > max_per_day:
                elig = elig[:max_per_day]
            for s in elig:
                all_eligible.append((d, s))

        unique_tickers = list(dict.fromkeys([s.ticker for _, s in all_eligible]))
        print(f"Total Eligible News Signals: {len(all_eligible)} | Unique Tickers: {len(unique_tickers)}")

        # Batch download market data including SPY and QQQ benchmarks
        all_syms = list(unique_tickers) + ["SPY", "QQQ"]
        start_dt = pd.to_datetime(date_list[0]) - timedelta(days=14)
        end_dt = pd.to_datetime(date_list[-1]) + timedelta(days=2)
        print(f"Downloading market data for {len(all_syms)} tickers (including SPY & QQQ benchmarks)...")

        ticker_data_map: Dict[str, pd.DataFrame] = {}
        try:
            raw_batch = yf.download(
                all_syms,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="15m",
                group_by="ticker",
                progress=False
            )
            for ticker in all_syms:
                if len(all_syms) == 1:
                    sub = raw_batch.copy()
                elif hasattr(raw_batch.columns, "levels") and ticker in raw_batch.columns.levels[0]:
                    sub = raw_batch[ticker].dropna(how="all").copy()
                elif ticker in raw_batch:
                    sub = raw_batch[ticker].dropna(how="all").copy()
                else:
                    continue

                if not sub.empty:
                    sub.columns = [c.lower() for c in sub.columns]
                    if sub.index.tz is not None:
                        sub.index = sub.index.tz_convert("America/New_York")
                    else:
                        sub.index = sub.index.tz_localize("UTC").tz_convert("America/New_York")
                    ticker_data_map[ticker] = sub
        except Exception as e:
            print(f"Batch download warning: {e}. Falling back to sequential download.")
            for ticker in all_syms:
                try:
                    df = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="15m", progress=False)
                    if not df.empty:
                        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert("America/New_York")
                        else:
                            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
                        ticker_data_map[ticker] = df
                except Exception:
                    pass

        print(f"Data ready for {len(ticker_data_map)} assets. Extracting full continuous trading schedule...")

        # Extract all continuous trading days in period from SPY or first asset
        all_market_dates = []
        ref_df = ticker_data_map.get("SPY", next(iter(ticker_data_map.values()), None))
        if ref_df is not None and not ref_df.empty:
            period_df = ref_df[(ref_df.index.strftime("%Y-%m-%d") >= date_list[0]) & (ref_df.index.strftime("%Y-%m-%d") <= date_list[-1])]
            all_market_dates = sorted(list(set(period_df.index.strftime("%Y-%m-%d"))))
        if not all_market_dates:
            all_market_dates = sorted(date_list)

        print(f"Total Market Trading Sessions in period: {len(all_market_dates)} days.")
        print(f"Running simulations across 3 technical engines + Buy & Hold benchmarks...")

        results = []
        # Run 1: Market Reactor Baseline
        res_mr = simulate_strategy_engine(
            "MarketReactor", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio
        )
        results.append(res_mr)

        # Run 2: News + MultiTimeframe Key Levels & Fib
        res_mtf = simulate_strategy_engine(
            "News_MTF_KeyLevels", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio
        )
        results.append(res_mtf)

        # Run 3: News + Liquidity Sweep
        res_swp = simulate_strategy_engine(
            "News_LiquiditySweep", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio
        )
        results.append(res_swp)

        # Calculate Buy & Hold Benchmarks ($100k allocation each)
        benchmarks = []
        for b_sym, b_name in [("SPY", "SPY (S&P 500 ETF)"), ("QQQ", "NASDAQ 100 (QQQ ETF)")]:
            if b_sym in ticker_data_map and not ticker_data_map[b_sym].empty:
                b_df = ticker_data_map[b_sym]
                p_df = b_df[(b_df.index.strftime("%Y-%m-%d") >= date_list[0]) & (b_df.index.strftime("%Y-%m-%d") <= date_list[-1])]
                if not p_df.empty:
                    s_price = float(p_df.iloc[0]["open"])
                    e_price = float(p_df.iloc[-1]["close"])
                    b_ret = (e_price - s_price) / s_price
                    b_pnl = initial_portfolio * b_ret
                    benchmarks.append({
                        "symbol": b_sym,
                        "name": b_name,
                        "start_price": s_price,
                        "end_price": e_price,
                        "return_pct": b_ret,
                        "pnl": b_pnl
                    })

        # Format side-by-side performance table
        print("\n" + "=" * 105)
        print(f" SIDE-BY-SIDE PERFORMANCE MATRIX — {period_title.upper()}")
        print("=" * 105)

        summary_rows = []
        for r in results:
            summary_rows.append({
                "Strategy / Benchmark": r["strategy"],
                "Total Trades": r["total_trades"],
                "Closed": r["closed_trades"],
                "Open": r["open_trades"],
                "Win Rate": f"{r['win_rate']:.1%}",
                "Profit Factor": f"{r['profit_factor']:.2f}",
                "Closed P&L": f"${r['closed_pnl']:+,.2f}",
                "Open P&L": f"${r['open_pnl']:+,.2f}",
                "Total Net P&L": f"${r['net_pnl']:+,.2f}",
                "Return %": f"{r['return_pct']:+.2%}"
            })

        for b in benchmarks:
            summary_rows.append({
                "Strategy / Benchmark": f"Buy & Hold: {b['name']}",
                "Total Trades": 1,
                "Closed": 0,
                "Open": 1,
                "Win Rate": "100.0%" if b["pnl"] > 0 else "0.0%",
                "Profit Factor": "N/A",
                "Closed P&L": "$0.00",
                "Open P&L": f"${b['pnl']:+,.2f}",
                "Total Net P&L": f"${b['pnl']:+,.2f}",
                "Return %": f"{b['return_pct']:+.2%}"
            })

        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.to_string(index=False))

        # Alpha Outperformance Summary
        if benchmarks:
            spy_b = next((b for b in benchmarks if b["symbol"] == "SPY"), None)
            qqq_b = next((b for b in benchmarks if b["symbol"] == "QQQ"), None)
            print("\n" + "-" * 105)
            print(" OUTPERFORMANCE (ALPHA) RELATIVE TO $100K BENCHMARKS:")
            print("-" * 105)
            for r in results:
                strat_ret = r["return_pct"]
                strat_pnl = r["net_pnl"]
                spy_alpha = (strat_ret - spy_b["return_pct"]) if spy_b else 0.0
                qqq_alpha = (strat_ret - qqq_b["return_pct"]) if qqq_b else 0.0
                spy_diff_pnl = (strat_pnl - spy_b["pnl"]) if spy_b else 0.0
                qqq_diff_pnl = (strat_pnl - qqq_b["pnl"]) if qqq_b else 0.0
                print(f" • {r['strategy']:<25}: vs SPY: {spy_alpha:+.2%} (${spy_diff_pnl:+,.2f}) | vs NASDAQ: {qqq_alpha:+.2%} (${qqq_diff_pnl:+,.2f})")

        # Save to logs
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        out_csv = os.path.join(BASE_DIR, "logs", f"comparative_news_strategies_{date_list[0]}_to_{date_list[-1]}.csv")
        df_summary.to_csv(out_csv, index=False)
        print(f"\n[OK] Comparative summary saved to: {out_csv}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "month"
    run_comparative_backtest(period_mode=mode)
