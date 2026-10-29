"""
Combined News Strategies Backtest Runner (run_combined_news_backtest.py)
========================================================================
Runs comparative multi-day portfolio backtests for:
1. Market Reactor - Dynamic (ORB + Gap Guard + Squeeze & Fade)
2. Market Reactor - Momentum Mode (ORB Squeeze Only)
3. Market Reactor - Fade Mode (Fundamental Reversal Only)
4. News + MultiTimeframe Key Levels & Fibonacci
5. News + Liquidity Sweep Engine
6. Benchmarks: SPY & NASDAQ-100 (QQQ) Buy & Hold ($100k Allocation)

Key Institutional Execution Upgrades:
- 15-Minute Opening Range Breakout (ORB) (9:30 - 9:45 AM EST).
- Daily Gap-to-ATR Exhaustion Guard (> 1.5x Daily ATR).
- Selector 1 (1H Swing, +1.5R breakeven ratchet, overnight allowed).
- Selector 2 (15m Intraday, strict 15:55 EST flatten, momentum/fade modes).
- Selector 3 (15m Intraday, conditional overnight hold if >= +0.5R at 15:50 EST).
- Conviction-Weighted Risk (1.0% to 2.5%, halved on exhausted gap).
- Order cancellation before flatten to eliminate orphan fills.
- Dual-cohort zero-lookahead for after-hours (> 16:00 EST) and intraday signals.
"""

import os
import sys
import json
import logging
from math import floor, sqrt
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
    base_risk_pct: float = 0.015,
    max_risk_pct: float = 0.025,
    selector2_mode: str = "dynamic"  # "dynamic", "momentum", "fade"
) -> Dict[str, Any]:
    """
    Simulates a specific technical engine gated by news direction over multiple days.
    Accurately carries active positions overnight across all continuous market trading sessions.
    """
    sorted_dates = sorted(all_market_dates) if all_market_dates else sorted(date_list)
    active_positions: Dict[str, Dict[str, Any]] = {}
    completed_trades: List[Dict[str, Any]] = []

    for d_str in sorted_dates:
        curr_date = pd.to_datetime(d_str).date()
        daily_sigs = [s for d, s in all_eligible_signals if d == d_str]

        # Filter out signals created after 16:00 EST on d_str (they belong to next session)
        session_sigs = []
        for sig in daily_sigs:
            if sig.created_at is not None and sig.signal_date == d_str and sig.created_at.time() >= time(16, 0):
                continue
            session_sigs.append(sig)

        # ── 1. ENTRY EVALUATION ACROSS NEW CANDLES ON CURR_DATE ──────────────
        for sig in session_sigs:
            ticker = sig.ticker
            if ticker in active_positions or ticker not in ticker_data_map:
                continue

            df = ticker_data_map[ticker]
            day_candles = df[df.index.date == curr_date]
            if len(day_candles) < 2:
                continue

            # First candle (09:30 - 09:45) establishes 15m Opening Range
            bar0 = day_candles.iloc[0]
            open_price = float(bar0["open"])
            or_high = float(bar0["high"])
            or_low = float(bar0["low"])

            # Compute Gap-to-ATR Exhaustion Guard
            prior_bars = df[df.index.date < curr_date]
            prev_close = float(prior_bars.iloc[-1]["close"]) if not prior_bars.empty else open_price
            daily_atr = calculate_atr(prior_bars, length=14) or (open_price * 0.02)
            gap_ratio = abs(open_price - prev_close) / daily_atr if daily_atr > 0 else 0.0
            is_exhausted_gap = gap_ratio > 1.5

            # Conviction-weighted position sizing (1.5% to 2.5%, halved if gap is exhausted)
            conv_score = getattr(sig, "composite_conviction", sqrt(max(0.0, sig.fundamental_confidence) * max(0.0, sig.news_conviction)))
            conv_norm = max(0.0, min(1.0, (conv_score - 0.70) / 0.30))
            eff_risk = base_risk_pct + conv_norm * (max_risk_pct - base_risk_pct)
            if is_exhausted_gap:
                eff_risk *= 0.50

            vol_ma = day_candles["volume"].rolling(20).mean()

            # ── ENGINE A: MARKET REACTOR (UPGRADED SELECTORS 1, 2, 3) ────────
            if strategy_name == "MarketReactor":
                for idx in range(1, len(day_candles)):
                    bar = day_candles.iloc[idx]
                    t = bar.name
                    bar_time = t.time()
                    if bar_time >= time(15, 30):
                        break

                    # Zero-lookahead: intraday catalyst must have arrived by bar_time
                    if sig.created_at is not None and sig.signal_date == d_str and bar_time < sig.created_at.time():
                        continue

                    p = float(bar["close"])
                    v = float(bar["volume"])
                    avg_v = vol_ma.iloc[idx] if not np.isnan(vol_ma.iloc[idx]) else v
                    rvol = (v / avg_v) if avg_v > 0 else 1.5

                    prior_df = df[df.index < t]
                    atr_val = calculate_atr(prior_df, length=14) or (open_price * 0.015)

                    tf_bars = prior_df.tail(40 if sig.selector_id == 1 else 16)
                    sup = float(tf_bars["low"].min()) if not tf_bars.empty else (or_low if or_low < p else p - (1.5 * atr_val))
                    res = float(tf_bars["high"].max()) if not tf_bars.empty else p + (2.5 * atr_val)

                    allowed_side = sig.signal_position  # "BUY" or "SELL" (news direction)
                    entry_triggered = False
                    resolved_side = allowed_side

                    # Selector 1: Macro Alignment (News + Fundamentals Agree)
                    if sig.selector_id == 1:
                        if allowed_side == "BUY":
                            if p >= or_high:
                                entry_triggered = True
                                resolved_side = "BUY"
                            elif not is_exhausted_gap and p > or_low and (p - or_low) <= (0.5 * atr_val):
                                entry_triggered = True
                                resolved_side = "BUY"
                        else:  # SELL
                            if p <= or_low:
                                entry_triggered = True
                                resolved_side = "SELL"
                            elif not is_exhausted_gap and p < or_high and (or_high - p) <= (0.5 * atr_val):
                                entry_triggered = True
                                resolved_side = "SELL"

                    # Selector 2: Catalyst Momentum vs. Fundamental Fade
                    elif sig.selector_id == 2:
                        fund_dir = "SELL" if allowed_side == "BUY" else "BUY"

                        # Mode A: Momentum Squeeze
                        if selector2_mode in ("momentum", "dynamic"):
                            if allowed_side == "BUY" and p > or_high and rvol >= 1.8 and not is_exhausted_gap:
                                entry_triggered = True
                                resolved_side = "BUY"
                            elif allowed_side == "SELL" and p < or_low and rvol >= 1.8 and not is_exhausted_gap:
                                entry_triggered = True
                                resolved_side = "SELL"

                        # Mode B: Fundamental Fade
                        if not entry_triggered and selector2_mode in ("fade", "dynamic"):
                            if allowed_side == "BUY" and p < or_low:
                                entry_triggered = True
                                resolved_side = "SELL"  # fade failed news into fundamental SELL
                            elif allowed_side == "SELL" and p > or_high:
                                entry_triggered = True
                                resolved_side = "BUY"  # fade into fundamental BUY

                    # Selector 3: Directional Flow (Neutral Fundamentals)
                    else:
                        if allowed_side == "BUY" and p >= or_high and rvol >= 1.3:
                            entry_triggered = True
                            resolved_side = "BUY"
                        elif allowed_side == "SELL" and p <= or_low and rvol >= 1.3:
                            entry_triggered = True
                            resolved_side = "SELL"

                    if entry_triggered:
                        side_str = "buy" if resolved_side == "BUY" else "sell"
                        raw_sl = sup if resolved_side == "BUY" else res
                        raw_tp = res if resolved_side == "BUY" else sup

                        trading_style = getattr(sig, "trading_style", "swing_trading" if sig.selector_id == 1 else "day_trading")
                        rl = optimize_sl_tp(
                            entry_price=p,
                            stop_loss=raw_sl,
                            take_profit=raw_tp,
                            side=side_str,
                            atr=atr_val,
                            trading_style=trading_style,
                            risk_reward_ratio=1.5
                        )
                        sl_price, tp_price = float(rl.stop_loss), float(rl.take_profit)
                        risk_per_share = abs(p - sl_price)

                        qty = floor((initial_portfolio * eff_risk) / max(0.01, risk_per_share))
                        qty = max(1, min(qty, floor((initial_portfolio * 0.95) / p)))

                        active_positions[ticker] = {
                            "ticker": ticker,
                            "position": resolved_side,
                            "entry_date": d_str,
                            "entry_time": t,
                            "entry_price": p,
                            "quantity": qty,
                            "stop_loss": sl_price,
                            "take_profit": tp_price,
                            "current_price": p,
                            "selector_id": sig.selector_id,
                            "initial_risk": risk_per_share,
                            "breakeven_ratchet": False
                        }
                        break

            # ── ENGINE B: NEWS + MULTI-TIMEFRAME KEY LEVELS & FIBONACCI ──────
            elif strategy_name == "News_MTF_KeyLevels":
                open_time = bar0.name
                prior_df = df[df.index < open_time]
                if len(prior_df) < 20:
                    continue

                atr_val = calculate_atr(prior_df, length=14) or (open_price * 0.015)
                atr_buf = atr_val * 1.0  # 1.0x 15m ATR buffer matching updated strategy

                df_1d = prior_df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
                df_1h = prior_df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
                df_15m = prior_df.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

                found_levels = []
                for tf_code, r_df in [("1D", df_1d), ("1H", df_1h), ("15m", df_15m)]:
                    if len(r_df) >= 10:
                        l = find_key_levels(r_df, resolution=tf_code)
                        if not l.empty:
                            found_levels.append(l)

                merged_l = merge_key_levels(pd.concat(found_levels, ignore_index=True)) if found_levels else pd.DataFrame()
                all_levels = np.sort(merged_l["level_price"].unique()) if not merged_l.empty else np.array([])
                sups = merged_l[merged_l["type"] == "support"].sort_values("level_price", ascending=False) if not merged_l.empty else pd.DataFrame()
                resists = merged_l[merged_l["type"] == "resistance"].sort_values("level_price", ascending=True) if not merged_l.empty else pd.DataFrame()

                # Fibonacci setups across 1D, 1H, 15m
                fib_list = []
                for tf_label, r_df in [("1D", df_1d), ("1H", df_1h), ("15m", df_15m)]:
                    if len(r_df) >= 15:
                        s = get_fibonacci_trade_setups(r_df, resolution=tf_label, backcandles=min(35, len(r_df) - 2))
                        if not s.empty:
                            fib_list.append(s)
                fib_setups = pd.concat(fib_list, ignore_index=True) if fib_list else pd.DataFrame()

                allowed_side = sig.signal_position
                for idx in range(1, len(day_candles)):
                    bar = day_candles.iloc[idx]
                    prev_b = day_candles.iloc[idx - 1]
                    t = bar.name
                    p = float(bar["close"])
                    b_open = float(bar["open"])
                    prev_close = float(prev_b["close"])

                    is_bullish = (p > b_open) and (p >= prev_close)
                    is_bearish = (p < b_open) and (p <= prev_close)

                    # Opening Range boundary guard: never buy below or_low or short above or_high
                    if allowed_side == "BUY" and or_low is not None and p < or_low:
                        continue
                    if allowed_side == "SELL" and or_high is not None and p > or_high:
                        continue

                    v = float(bar["volume"])
                    avg_v = vol_ma.iloc[idx] if not np.isnan(vol_ma.iloc[idx]) else v
                    rvol = (v / avg_v) if avg_v > 0 else 1.3

                    entry_setup = None

                    # Setup 1: MTF S/R Flip Breakout (RVOL >= 1.25)
                    if len(all_levels) > 0 and rvol >= 1.25:
                        if allowed_side == "BUY" and is_bullish:
                            below = all_levels[all_levels <= p]
                            above = all_levels[all_levels > p + (0.5 * atr_val)]
                            if len(below) > 0:
                                flipped = float(below[-1])
                                if 0 <= (p - flipped) <= (flipped * 0.008):
                                    sl = flipped - atr_buf
                                    tp = float(above[0]) if len(above) > 0 else p + (2.5 * atr_val)
                                    if (tp - p) / max(0.01, p - sl) >= 1.6:
                                        entry_setup = ("BUY", p, sl, tp, "MTF_SR_FLIP_BREAKOUT")
                        elif allowed_side == "SELL" and is_bearish:
                            above = all_levels[all_levels >= p]
                            below = all_levels[all_levels < p - (0.5 * atr_val)]
                            if len(above) > 0:
                                flipped = float(above[0])
                                if 0 <= (flipped - p) <= (flipped * 0.008):
                                    sl = flipped + atr_buf
                                    tp = float(below[-1]) if len(below) > 0 else p - (2.5 * atr_val)
                                    if (p - tp) / max(0.01, sl - p) >= 1.6:
                                        entry_setup = ("SELL", p, sl, tp, "MTF_SR_FLIP_BREAKDOWN")

                    # Setup 2: Confirmed Fibonacci 0.618 Retracement
                    if entry_setup is None and not fib_setups.empty:
                        for _, f_row in fib_setups.iterrows():
                            f_side = "BUY" if str(f_row["trend"]).lower() == "uptrend" else "SELL"
                            if f_side != allowed_side:
                                continue
                            f_entry = float(f_row["entry_price"])
                            if abs(p - f_entry) <= (f_entry * 0.006):
                                if f_side == "BUY" and is_bullish and p >= f_entry:
                                    raw_sl = min(float(f_row["stop_loss"]), p - atr_buf)
                                    raw_tp = float(f_row["take_profit"])
                                    if raw_tp > p and (raw_tp - p) / max(0.01, p - raw_sl) >= 1.6:
                                        entry_setup = ("BUY", p, raw_sl, raw_tp, "FIB_0.618_CONFIRMED")
                                        break
                                elif f_side == "SELL" and is_bearish and p <= f_entry:
                                    raw_sl = max(float(f_row["stop_loss"]), p + atr_buf)
                                    raw_tp = float(f_row["take_profit"])
                                    if raw_tp < p and (p - raw_tp) / max(0.01, raw_sl - p) >= 1.6:
                                        entry_setup = ("SELL", p, raw_sl, raw_tp, "FIB_0.618_CONFIRMED")
                                        break

                    # Setup 3: Confirmed MTF Key Level Rejection
                    if entry_setup is None:
                        if allowed_side == "BUY" and not sups.empty and is_bullish:
                            valid_sups = sups[sups["level_price"] <= p * 1.003]["level_price"].values
                            valid_ress = resists[resists["level_price"] > p + atr_val]["level_price"].values
                            for s_px in valid_sups:
                                s_val = float(s_px)
                                if abs(p - s_val) <= (s_val * 0.006) and p > s_val:
                                    sl = s_val - atr_buf
                                    tp = float(valid_ress[0]) if len(valid_ress) > 0 else p + (2.5 * atr_val)
                                    if (tp - p) / max(0.01, p - sl) >= 1.6:
                                        entry_setup = ("BUY", p, sl, tp, "MTF_SUPPORT_REJECTION")
                                        break
                        elif allowed_side == "SELL" and not resists.empty and is_bearish:
                            valid_ress = resists[resists["level_price"] >= p * 0.997]["level_price"].values
                            valid_sups = sups[sups["level_price"] < p - atr_val]["level_price"].values
                            for r_px in valid_ress:
                                r_val = float(r_px)
                                if abs(p - r_val) <= (r_val * 0.006) and p < r_val:
                                    sl = r_val + atr_buf
                                    tp = float(valid_sups[-1]) if len(valid_sups) > 0 else p - (2.5 * atr_val)
                                    if (p - tp) / max(0.01, sl - p) >= 1.6:
                                        entry_setup = ("SELL", p, sl, tp, "MTF_RESISTANCE_REJECTION")
                                        break

                    if entry_setup is not None:
                        e_side, e_p, e_sl, e_tp, e_reason = entry_setup
                        risk_per_share = abs(e_p - e_sl)
                        qty = floor((initial_portfolio * eff_risk) / max(0.01, risk_per_share))
                        qty = max(1, min(qty, floor((initial_portfolio * 0.95) / e_p)))

                        active_positions[ticker] = {
                            "ticker": ticker, "position": e_side, "entry_date": d_str,
                            "entry_time": t, "entry_price": e_p, "quantity": qty,
                            "stop_loss": e_sl, "take_profit": e_tp, "current_price": e_p,
                            "selector_id": 1, "initial_risk": risk_per_share, "breakeven_ratchet": False,
                            "be_trigger_r": 1.2
                        }
                        break

            # ── ENGINE C: NEWS + LIQUIDITY SWEEP (MULTI-POOL ENGINE) ────────
            elif strategy_name == "News_LiquiditySweep":
                open_time = bar0.name
                prior_dates = sorted(list(set(df[df.index.date < curr_date].index.date)))
                if not prior_dates:
                    continue

                last_d = prior_dates[-1]
                prev_day_bars = df[df.index.date == last_d]
                if prev_day_bars.empty:
                    continue

                pdh = float(prev_day_bars["high"].max())
                pdl = float(prev_day_bars["low"].min())
                pdc = float(prev_day_bars["close"].iloc[-1])
                atr_val = calculate_atr(df[df.index <= open_time], length=14) or (open_price * 0.015)
                atr_buf = atr_val * 0.8  # 0.8x 15m ATR buffer beyond sweep wick

                allowed_side = sig.signal_position

                # Multi-pool levels: Pool 1 (ORB Extreme), Pool 2 (PDC), Pool 3 (PDH/PDL)
                target_pools = [pdl, pdc, or_low] if allowed_side == "BUY" else [pdh, pdc, or_high]

                for idx in range(1, len(day_candles)):
                    bar = day_candles.iloc[idx]
                    h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
                    v = float(bar["volume"])
                    v_spike = v >= (vol_ma.iloc[idx] * 1.15) if not np.isnan(vol_ma.iloc[idx]) else True
                    t = bar.name

                    sweep_trigger = None

                    if allowed_side == "BUY" and v_spike:
                        for pool_lvl in target_pools:
                            if pool_lvl is not None and l < pool_lvl and c > pool_lvl:
                                sl_price = l - atr_buf
                                tp_price = max(pdh, c + (2.0 * atr_val))
                                if (tp_price - c) / max(0.01, c - sl_price) >= 1.6:
                                    sweep_trigger = ("BUY", c, sl_price, tp_price, f"SWEEP_RECLAIM_{pool_lvl:.2f}")
                                    break

                    elif allowed_side == "SELL" and v_spike:
                        for pool_lvl in target_pools:
                            if pool_lvl is not None and h > pool_lvl and c < pool_lvl:
                                sl_price = h + atr_buf
                                tp_price = min(pdl, c - (2.0 * atr_val))
                                if (c - tp_price) / max(0.01, sl_price - c) >= 1.6:
                                    sweep_trigger = ("SELL", c, sl_price, tp_price, f"SWEEP_RECLAIM_{pool_lvl:.2f}")
                                    break

                    if sweep_trigger is not None:
                        e_side, e_p, e_sl, e_tp, e_reason = sweep_trigger
                        risk_per_share = abs(e_p - e_sl)
                        qty = floor((initial_portfolio * eff_risk) / max(0.01, risk_per_share))
                        qty = max(1, min(qty, floor((initial_portfolio * 0.95) / e_p)))

                        active_positions[ticker] = {
                            "ticker": ticker, "position": e_side, "entry_date": d_str,
                            "entry_time": t, "entry_price": e_p, "quantity": qty,
                            "stop_loss": e_sl, "take_profit": e_tp, "current_price": e_p,
                            "selector_id": 2, "initial_risk": risk_per_share, "breakeven_ratchet": False,
                            "be_trigger_r": 1.2
                        }
                        break

        # ── 2. MONITOR ACTIVE POSITIONS ACROSS CURR_DATE CANDLES ─────────────
        closed_tickers = []
        for ticker, pos in active_positions.items():
            if ticker not in ticker_data_map:
                continue
            eval_candles = ticker_data_map[ticker][(ticker_data_map[ticker].index.date == curr_date) & (ticker_data_map[ticker].index >= pos["entry_time"])]

            for b_idx in range(len(eval_candles)):
                bar = eval_candles.iloc[b_idx]
                h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
                pos["current_price"] = c
                b_time = bar.name.time()

                # Breakeven Stop Ratchet (+1.2R for MTF/Sweep, +1.5R for Selector 1)
                be_r = pos.get("be_trigger_r", 1.5)
                if not pos.get("breakeven_ratchet", False):
                    signed_gain = (c - pos["entry_price"]) if pos["position"] == "BUY" else (pos["entry_price"] - c)
                    r_mult = signed_gain / pos["initial_risk"] if pos["initial_risk"] > 0 else 0.0
                    if r_mult >= be_r:
                        pos["stop_loss"] = pos["entry_price"]
                        pos["breakeven_ratchet"] = True

                # Selector 2: Strict Intraday Flatten at 15:55 EST (15:45 bar close) on entry date
                if pos.get("selector_id") == 2 and pos["entry_date"] == d_str and b_time >= time(15, 45):
                    pos["status"] = "EOD_SEL2_FLATTEN"
                    pos["exit_price"] = c
                    pos["exit_time"] = bar.name
                    pos["pnl"] = (c - pos["entry_price"]) * pos["quantity"] if pos["position"] == "BUY" else (pos["entry_price"] - c) * pos["quantity"]
                    pos["pnl_pct"] = (c - pos["entry_price"]) / pos["entry_price"] if pos["position"] == "BUY" else (pos["entry_price"] - c) / pos["entry_price"]
                    completed_trades.append(pos)
                    closed_tickers.append(ticker)
                    break

                # Selector 3: Conditional Overnight Flatten at 15:50 EST (15:45 bar close) if < +0.5R
                if pos.get("selector_id") == 3 and pos["entry_date"] == d_str and b_time >= time(15, 45):
                    signed_gain = (c - pos["entry_price"]) if pos["position"] == "BUY" else (pos["entry_price"] - c)
                    r_mult = signed_gain / pos["initial_risk"] if pos["initial_risk"] > 0 else 0.0
                    if r_mult < 0.50:
                        pos["status"] = "EOD_SEL3_FLATTEN"
                        pos["exit_price"] = c
                        pos["exit_time"] = bar.name
                        pos["pnl"] = signed_gain * pos["quantity"]
                        pos["pnl_pct"] = signed_gain / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break

                # TP & SL Checks
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
                        pos["status"] = "STOPPED OUT" if not pos.get("breakeven_ratchet") else "BREAKEVEN EXIT"
                        pos["exit_price"] = pos["stop_loss"]
                        pos["exit_time"] = bar.name
                        pos["pnl"] = (pos["stop_loss"] - pos["entry_price"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["stop_loss"] - pos["entry_price"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break
                else:  # SELL
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
                        pos["status"] = "STOPPED OUT" if not pos.get("breakeven_ratchet") else "BREAKEVEN EXIT"
                        pos["exit_price"] = pos["stop_loss"]
                        pos["exit_time"] = bar.name
                        pos["pnl"] = (pos["entry_price"] - pos["stop_loss"]) * pos["quantity"]
                        pos["pnl_pct"] = (pos["entry_price"] - pos["stop_loss"]) / pos["entry_price"]
                        completed_trades.append(pos)
                        closed_tickers.append(ticker)
                        break

        # Cancel orders and remove closed positions to prevent orphan order fills
        for t in closed_tickers:
            if t in active_positions:
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
    sl_hits = len(df_c[df_c["status"].str.contains("STOP", na=False)]) if not df_c.empty else 0
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


def run_comparative_backtest(period_mode: str = "2months"):
    periods = []
    if period_mode in ("2months", "2month", "all"):
        two_month_dates = [
            "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-17",
            "2026-09-18", "2026-09-22", "2026-09-23", "2026-09-29",
            "2026-09-30", "2026-10-01", "2026-10-13", "2026-10-14",
            "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20",
            "2026-10-28", "2026-10-29"
        ]
        periods.append(("Full 2-Month Evaluation (Sep 11 - Oct 29, 2026)", two_month_dates, 8))
    if period_mode in ("month", "1month", "both", "all"):
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

        # Ingest eligible signals including prior after-hours
        all_eligible: List[Tuple[str, ReactorSignal]] = []
        for d in date_list:
            sigs = fetch_reactor_signals(d, include_prev_afterhours=True)
            elig = [s for s in sigs if s.is_eligible]
            elig.sort(
                key=lambda x: getattr(x, "composite_conviction", x.fundamental_confidence * x.news_conviction),
                reverse=True
            )
            if max_per_day and len(elig) > max_per_day:
                elig = elig[:max_per_day]
            for s in elig:
                all_eligible.append((d, s))

        unique_tickers = list(dict.fromkeys([s.ticker for _, s in all_eligible]))
        print(f"Total Eligible News Signals: {len(all_eligible)} | Unique Tickers: {len(unique_tickers)}")

        # Batch download market data (extends 5 trading days after date_list[-1] for multi-day runners)
        # Yahoo Finance limits 15m intraday data to the last 60 calendar days
        all_syms = list(unique_tickers) + ["SPY", "QQQ"]
        earliest_safe = pd.Timestamp.now().tz_localize(None) - timedelta(days=58)
        start_dt = max(pd.to_datetime(date_list[0]) - timedelta(days=10), earliest_safe)
        end_dt = pd.to_datetime(date_list[-1]) + timedelta(days=5)
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

        # Extract continuous trading days
        all_market_dates = []
        ref_df = ticker_data_map.get("SPY", next(iter(ticker_data_map.values()), None))
        if ref_df is not None and not ref_df.empty:
            period_df = ref_df[(ref_df.index.strftime("%Y-%m-%d") >= date_list[0]) & (ref_df.index.strftime("%Y-%m-%d") <= date_list[-1])]
            all_market_dates = sorted(list(set(period_df.index.strftime("%Y-%m-%d"))))
        if not all_market_dates:
            all_market_dates = sorted(date_list)

        print(f"Total Market Trading Sessions in period: {len(all_market_dates)} days.")
        print(f"Running comparative simulations across Selector 2 modes + Technical Engines + Benchmarks...")

        results = []

        # 1. Market Reactor - Dynamic Mode (ORB Squeeze + Fundamental Fade)
        res_mr_dyn = simulate_strategy_engine(
            "MarketReactor", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio,
            selector2_mode="dynamic"
        )
        res_mr_dyn["strategy"] = "MarketReactor (Dynamic)"
        results.append(res_mr_dyn)

        # 2. Market Reactor - Momentum Mode (ORB Squeeze Only)
        res_mr_mom = simulate_strategy_engine(
            "MarketReactor", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio,
            selector2_mode="momentum"
        )
        res_mr_mom["strategy"] = "MarketReactor (Momentum)"
        results.append(res_mr_mom)

        # 3. Market Reactor - Fade Mode (Fundamental Reversal Only)
        res_mr_fade = simulate_strategy_engine(
            "MarketReactor", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio,
            selector2_mode="fade"
        )
        res_mr_fade["strategy"] = "MarketReactor (Fade)"
        results.append(res_mr_fade)

        # 4. News + MultiTimeframe Key Levels & Fib
        res_mtf = simulate_strategy_engine(
            "News_MTF_KeyLevels", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio
        )
        results.append(res_mtf)

        # 5. News + Liquidity Sweep
        res_swp = simulate_strategy_engine(
            "News_LiquiditySweep", date_list, all_eligible, ticker_data_map,
            all_market_dates=all_market_dates, initial_portfolio=initial_portfolio
        )
        results.append(res_swp)

        # 6. Calculate Buy & Hold Benchmarks ($100k allocation each)
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
        print("\n" + "=" * 115)
        print(f" SIDE-BY-SIDE PERFORMANCE MATRIX — {period_title.upper()}")
        print("=" * 115)

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
            print("\n" + "-" * 115)
            print(" OUTPERFORMANCE (ALPHA) RELATIVE TO $100K BENCHMARKS:")
            print("-" * 115)
            for r in results:
                strat_ret = r["return_pct"]
                strat_pnl = r["net_pnl"]
                spy_alpha = (strat_ret - spy_b["return_pct"]) if spy_b else 0.0
                qqq_alpha = (strat_ret - qqq_b["return_pct"]) if qqq_b else 0.0
                spy_diff_pnl = (strat_pnl - spy_b["pnl"]) if spy_b else 0.0
                qqq_diff_pnl = (strat_pnl - qqq_b["pnl"]) if qqq_b else 0.0
                print(f" • {r['strategy']:<27}: vs SPY: {spy_alpha:+.2%} (${spy_diff_pnl:+,.2f}) | vs NASDAQ: {qqq_alpha:+.2%} (${qqq_diff_pnl:+,.2f})")

        # Save to logs
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        out_csv = os.path.join(BASE_DIR, "logs", f"comparative_news_strategies_{date_list[0]}_to_{date_list[-1]}.csv")
        df_summary.to_csv(out_csv, index=False)
        print(f"\n[OK] Comparative summary saved to: {out_csv}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "2months"
    run_comparative_backtest(period_mode=mode)
