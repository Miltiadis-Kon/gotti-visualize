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

            # ── ENGINE B: NEWS + CATALYST IMPULSE FIBONACCI & MTF CONFLUENCE (CIGC) ─
            elif strategy_name == "News_MTF_KeyLevels":
                open_time = bar0.name
                prior_df = df[df.index < open_time]
                if len(prior_df) < 15:
                    continue

                atr_val = calculate_atr(prior_df, length=14) or (open_price * 0.015)
                atr_buf = atr_val * 0.5  # 0.5x 15m ATR buffer beyond Fib/swing extreme

                # MTF Key Levels & PDC/PDH/PDL
                df_1d = prior_df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
                df_1h = prior_df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
                df_15m = prior_df.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

                pdc = float(df_1d.iloc[-1]["close"]) if not df_1d.empty else open_price
                pdh = float(df_1d.iloc[-1]["high"]) if not df_1d.empty else open_price * 1.01
                pdl = float(df_1d.iloc[-1]["low"]) if not df_1d.empty else open_price * 0.99

                found_levels = []
                for tf_code, r_df in [("1D", df_1d), ("1H", df_1h), ("15m", df_15m)]:
                    if len(r_df) >= 10:
                        l = find_key_levels(r_df, resolution=tf_code)
                        if not l.empty:
                            found_levels.append(l)

                merged_l = merge_key_levels(pd.concat(found_levels, ignore_index=True)) if found_levels else pd.DataFrame()
                all_levels = np.sort(merged_l["level_price"].unique()) if not merged_l.empty else np.array([])

                # Higher-timeframe macro Fibonacci swings
                fib_list = []
                for tf_label, r_df in [("1D", df_1d), ("1H", df_1h)]:
                    if len(r_df) >= 20:
                        s = get_fibonacci_trade_setups(r_df, resolution=tf_label, backcandles=min(35, len(r_df) - 2))
                        if not s.empty:
                            fib_list.append(s)
                htf_fib = pd.concat(fib_list, ignore_index=True) if fib_list else pd.DataFrame()

                # Intraday Indicators: EMA9, EMA20, VWAP
                ema9 = day_candles["close"].ewm(span=9, adjust=False).mean()
                ema20 = day_candles["close"].ewm(span=20, adjust=False).mean()
                typical_p = (day_candles["high"] + day_candles["low"] + day_candles["close"]) / 3.0
                cum_pv = (typical_p * day_candles["volume"]).cumsum()
                cum_vol = day_candles["volume"].cumsum()
                vwap = (cum_pv / cum_vol.replace(0, np.nan)).ffill().bfill()

                allowed_side = sig.signal_position
                conf_tol = 0.006

                def has_mtf_conf(px, v_val):
                    if abs(px - v_val) / max(0.01, v_val) <= conf_tol:
                        return True, "VWAP"
                    if abs(px - pdh) / max(0.01, pdh) <= conf_tol:
                        return True, "PDH_FLIP"
                    if abs(px - pdl) / max(0.01, pdl) <= conf_tol:
                        return True, "PDL_FLIP"
                    if len(all_levels) > 0 and any(abs(px - kl) / max(0.01, kl) <= conf_tol for kl in all_levels):
                        return True, "MTF_SR"
                    return False, "NONE"

                entry_setup = None
                thesis_invalidated = False

                for idx in range(1, len(day_candles)):
                    if thesis_invalidated:
                        break

                    bar = day_candles.iloc[idx]
                    prev_b = day_candles.iloc[idx - 1]
                    t = bar.name
                    p = float(bar["close"])
                    b_open = float(bar["open"])
                    b_high = float(bar["high"])
                    b_low = float(bar["low"])
                    prev_close = float(prev_b["close"])
                    v = float(bar["volume"])
                    avg_v = vol_ma.iloc[idx] if not np.isnan(vol_ma.iloc[idx]) else v

                    is_bullish = (p > b_open) and (p >= prev_close)
                    is_bearish = (p < b_open) and (p <= prev_close)

                    cur_vwap = float(vwap.iloc[idx])
                    cur_ema9 = float(ema9.iloc[idx])
                    cur_ema20 = float(ema20.iloc[idx])

                    catalyst_bars = day_candles.iloc[:idx+1]
                    if len(catalyst_bars) < 2:
                        continue

                    # LONG CATALYST FIBONACCI (BUY)
                    if allowed_side == "BUY":
                        peak_idx = catalyst_bars["high"].idxmax()
                        bars_up_to_peak = catalyst_bars.loc[:peak_idx]
                        bars_after_peak = catalyst_bars.loc[peak_idx:]

                        session_low_before_peak = float(bars_up_to_peak["low"].min())
                        anchor_b = float(catalyst_bars.loc[peak_idx, "high"])
                        anchor_a = max(pdc, session_low_before_peak - (1.0 * atr_val)) if session_low_before_peak > pdc else session_low_before_peak

                        impulse_size = anchor_b - anchor_a
                        if impulse_size < (0.9 * atr_val):
                            continue

                        fib_236 = anchor_b - (0.236 * impulse_size)
                        fib_382 = anchor_b - (0.382 * impulse_size)
                        fib_500 = anchor_b - (0.500 * impulse_size)
                        fib_618 = anchor_b - (0.618 * impulse_size)
                        fib_786 = anchor_b - (0.786 * impulse_size)
                        ext_1272 = anchor_b + (0.272 * impulse_size)
                        ext_1618 = anchor_b + (0.618 * impulse_size)

                        pullback_low = float(bars_after_peak["low"].min()) if len(bars_after_peak) > 1 else b_low
                        if pullback_low < fib_786 or p < fib_786:
                            thesis_invalidated = True
                            break

                        impulse_avg_vol = float(bars_up_to_peak["volume"].mean())
                        healthy_vol = (v <= impulse_avg_vol * 1.35) or is_bullish

                        if len(bars_after_peak) >= 2 and is_bullish and healthy_vol:
                            # Tier 1: 0.382 - 0.500 Shallow Reload
                            tested_shallow = (fib_500 * 0.998 <= pullback_low <= fib_236 * 1.001)
                            in_window = (fib_500 <= p <= anchor_b)
                            if tested_shallow and in_window and (p >= cur_vwap or p >= cur_ema20):
                                sl = min(pullback_low, fib_618) - atr_buf
                                tp = ext_1272 if (ext_1272 - p) / max(0.01, p - sl) >= 1.6 else ext_1618
                                if (tp - p) / max(0.01, p - sl) >= 1.6:
                                    entry_setup = ("BUY", p, sl, tp, anchor_b, "CIGC_TIER1_FIB_0.382_RELOAD")
                                    break

                            # Tier 2: 0.500 - 0.618 Golden Pocket + MTF/VWAP Confluence
                            tested_pocket = (fib_786 < pullback_low <= fib_500 * 1.002)
                            reclaimed_pocket = (fib_618 <= p <= fib_236) and (p >= cur_ema9)
                            has_conf, conf_tag = has_mtf_conf(fib_618, cur_vwap)
                            if tested_pocket and reclaimed_pocket and (has_conf or sig.selector_id == 1):
                                sl = min(pullback_low, fib_786) - atr_buf
                                tp = ext_1272 if (ext_1272 - p) / max(0.01, p - sl) >= 1.6 else ext_1618
                                if (tp - p) / max(0.01, p - sl) >= 1.6:
                                    entry_setup = ("BUY", p, sl, tp, anchor_b, f"CIGC_TIER2_GOLDEN_POCKET_{conf_tag}")
                                    break

                    # SHORT CATALYST FIBONACCI (SELL)
                    elif allowed_side == "SELL":
                        trough_idx = catalyst_bars["low"].idxmin()
                        bars_down_to_trough = catalyst_bars.loc[:trough_idx]
                        bars_after_trough = catalyst_bars.loc[trough_idx:]

                        session_high_before_trough = float(bars_down_to_trough["high"].max())
                        anchor_b = float(catalyst_bars.loc[trough_idx, "low"])
                        anchor_a = min(pdc, session_high_before_trough + (1.0 * atr_val)) if session_high_before_trough < pdc else session_high_before_trough

                        impulse_size = anchor_a - anchor_b
                        if impulse_size < (0.9 * atr_val):
                            continue

                        fib_236 = anchor_b + (0.236 * impulse_size)
                        fib_382 = anchor_b + (0.382 * impulse_size)
                        fib_500 = anchor_b + (0.500 * impulse_size)
                        fib_618 = anchor_b + (0.618 * impulse_size)
                        fib_786 = anchor_b + (0.786 * impulse_size)
                        ext_1272 = anchor_b - (0.272 * impulse_size)
                        ext_1618 = anchor_b - (0.618 * impulse_size)

                        pullback_high = float(bars_after_trough["high"].max()) if len(bars_after_trough) > 1 else b_high
                        if pullback_high > fib_786 or p > fib_786:
                            thesis_invalidated = True
                            break

                        impulse_avg_vol = float(bars_down_to_trough["volume"].mean())
                        healthy_vol = (v <= impulse_avg_vol * 1.35) or is_bearish

                        if len(bars_after_trough) >= 2 and is_bearish and healthy_vol:
                            # Tier 1: 0.382 - 0.500 Shallow Bearish Reload
                            tested_shallow = (fib_236 * 0.999 <= pullback_high <= fib_500 * 1.002)
                            in_window = (anchor_b <= p <= fib_500)
                            if tested_shallow and in_window and (p <= cur_vwap or p <= cur_ema20):
                                sl = max(pullback_high, fib_618) + atr_buf
                                tp = ext_1272 if (p - ext_1272) / max(0.01, sl - p) >= 1.6 else ext_1618
                                if (p - tp) / max(0.01, sl - p) >= 1.6:
                                    entry_setup = ("SELL", p, sl, tp, anchor_b, "CIGC_TIER1_SHORT_FIB_0.382")
                                    break

                            # Tier 2: 0.500 - 0.618 Bearish Golden Pocket
                            tested_pocket = (fib_500 * 0.998 <= pullback_high < fib_786)
                            reclaimed_pocket = (fib_236 <= p <= fib_618) and (p <= cur_ema9)
                            has_conf, conf_tag = has_mtf_conf(fib_618, cur_vwap)
                            if tested_pocket and reclaimed_pocket and (has_conf or sig.selector_id == 1):
                                sl = max(pullback_high, fib_786) + atr_buf
                                tp = ext_1272 if (p - ext_1272) / max(0.01, sl - p) >= 1.6 else ext_1618
                                if (p - tp) / max(0.01, sl - p) >= 1.6:
                                    entry_setup = ("SELL", p, sl, tp, anchor_b, f"CIGC_TIER2_SHORT_GOLDEN_POCKET_{conf_tag}")
                                    break

                if entry_setup is not None:
                    e_side, e_p, e_sl, e_tp, e_peak, e_reason = entry_setup
                    risk_per_share = abs(e_p - e_sl)
                    qty = floor((initial_portfolio * eff_risk) / max(0.01, risk_per_share))
                    qty = max(1, min(qty, floor((initial_portfolio * 0.95) / e_p)))

                    active_positions[ticker] = {
                        "ticker": ticker, "position": e_side, "entry_date": d_str,
                        "entry_time": t, "entry_price": e_p, "quantity": qty,
                        "stop_loss": e_sl, "take_profit": e_tp, "current_price": e_p,
                        "selector_id": sig.selector_id, "initial_risk": risk_per_share,
                        "breakeven_ratchet": False, "impulse_peak": e_peak,
                        "be_trigger_r": 1.1
                    }
                    break

            # ── ENGINE C: NEWS + LIQUIDITY SWEEP (MULTI-POOL INSTITUTIONAL ENGINE) ─
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

                # Multi-pool levels matching news_liquidity_sweep.py:
                # Core downside liquidity pools for BUY: PDL, PDC (Gap Fill), OR_LOW
                # Core upside liquidity pools for SELL: PDH, PDC (Gap Fill), OR_HIGH
                target_pools = [
                    ("PDL", pdl), ("PDC_GAP", pdc), ("OR_LOW", or_low)
                ] if allowed_side == "BUY" else [
                    ("PDH", pdh), ("PDC_GAP", pdc), ("OR_HIGH", or_high)
                ]

                # Deduplicate levels within 0.15%
                unique_pools = []
                for p_name, p_lvl in target_pools:
                    if p_lvl is not None and not any(abs(p_lvl - ex_lvl) / max(0.01, ex_lvl) < 0.0015 for _, ex_lvl in unique_pools):
                        unique_pools.append((p_name, p_lvl))

                for idx in range(1, len(day_candles)):
                    bar = day_candles.iloc[idx]
                    h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
                    v = float(bar["volume"])
                    v_spike = v >= (vol_ma.iloc[idx] * 1.15) if not np.isnan(vol_ma.iloc[idx]) else True
                    t = bar.name

                    sweep_trigger = None

                    if allowed_side == "BUY" and v_spike:
                        for pool_name, pool_lvl in unique_pools:
                            # 15m sweep & reclaim: wick sweeps below level and candle closes back above
                            if pool_lvl is not None and l < pool_lvl and c > pool_lvl:
                                valid_depth = (pool_lvl - l) <= (1.5 * atr_val)
                                if valid_depth:
                                    sl_price = l - atr_buf
                                    tp_price = max(pdh, c + (2.0 * atr_val))
                                    if (tp_price - c) / max(0.01, c - sl_price) >= 1.6:
                                        sweep_trigger = ("BUY", c, sl_price, tp_price, f"{pool_name}_RECLAIM")
                                        break

                    elif allowed_side == "SELL" and v_spike:
                        for pool_name, pool_lvl in unique_pools:
                            # 15m sweep & reclaim: wick sweeps above level and candle closes back below
                            if pool_lvl is not None and h > pool_lvl and c < pool_lvl:
                                valid_depth = (h - pool_lvl) <= (1.5 * atr_val)
                                if valid_depth:
                                    sl_price = h + atr_buf
                                    tp_price = min(pdl, c - (2.0 * atr_val))
                                    if (c - tp_price) / max(0.01, sl_price - c) >= 1.6:
                                        sweep_trigger = ("SELL", c, sl_price, tp_price, f"{pool_name}_RECLAIM")
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

                # Breakeven / Wave-3 Ratchet (+0.2R for CIGC Fib, Breakeven for Selector 1 / Sweep)
                if not pos.get("breakeven_ratchet", False):
                    peak = pos.get("impulse_peak")
                    reached_peak = (
                        peak is not None and (
                            (pos["position"] == "BUY" and c >= peak) or
                            (pos["position"] == "SELL" and c <= peak)
                        )
                    )
                    signed_gain = (c - pos["entry_price"]) if pos["position"] == "BUY" else (pos["entry_price"] - c)
                    r_mult = signed_gain / pos["initial_risk"] if pos["initial_risk"] > 0 else 0.0
                    be_r = pos.get("be_trigger_r", 1.2)

                    if reached_peak or r_mult >= be_r:
                        cushion = (0.20 * pos["initial_risk"]) if peak is not None else 0.0
                        pos["stop_loss"] = (pos["entry_price"] + cushion) if pos["position"] == "BUY" else (pos["entry_price"] - cushion)
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


def run_comparative_backtest(period_mode: str = "6months"):
    periods = []
    if period_mode in ("6months", "6month", "all"):
        six_month_dates = [
            "2026-08-29", "2026-09-11", "2026-09-14", "2026-09-15",
            "2026-09-17", "2026-09-18", "2026-09-20", "2026-09-22",
            "2026-09-23", "2026-09-29", "2026-09-30", "2026-10-01",
            "2026-10-03", "2026-10-13", "2026-10-14", "2026-10-15",
            "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-28",
            "2026-10-29"
        ]
        periods.append(("Full 6-Month Evaluation (Apr 29 - Oct 29, 2026)", six_month_dates, 8))
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
                        "name": f"{b_name} [Active Window]",
                        "start_price": s_price,
                        "end_price": e_price,
                        "return_pct": b_ret,
                        "pnl": b_pnl
                    })

        # Add Full 6-Month Macro Benchmark (April 29 - October 29) if evaluating 6-month period
        if "6-Month" in period_title or period_mode in ("6months", "6month"):
            try:
                macro_df = yf.download(["SPY", "QQQ"], start="2026-04-29", end="2026-10-30", interval="1d", progress=False)
                for b_sym, b_name in [("SPY", "SPY (S&P 500 ETF)"), ("QQQ", "NASDAQ 100 (QQQ ETF)")]:
                    if "Open" in macro_df and b_sym in macro_df["Open"]:
                        m_open = float(macro_df["Open"][b_sym].dropna().iloc[0])
                        m_close = float(macro_df["Close"][b_sym].dropna().iloc[-1])
                        m_ret = (m_close - m_open) / m_open
                        m_pnl = initial_portfolio * m_ret
                        benchmarks.append({
                            "symbol": f"{b_sym}_6M",
                            "name": f"{b_name} [Full 6-Month Macro]",
                            "start_price": m_open,
                            "end_price": m_close,
                            "return_pct": m_ret,
                            "pnl": m_pnl
                        })
            except Exception as e:
                print(f"Macro benchmark download notice: {e}")

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
            spy_b = next((b for b in benchmarks if b["symbol"] in ("SPY_6M", "SPY")), None)
            qqq_b = next((b for b in benchmarks if b["symbol"] in ("QQQ_6M", "QQQ")), None)
            print("\n" + "-" * 115)
            print(f" OUTPERFORMANCE (ALPHA) RELATIVE TO $100K BENCHMARKS ({spy_b['name'] if spy_b else 'SPY'} & {qqq_b['name'] if qqq_b else 'QQQ'}):")
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "6months"
    run_comparative_backtest(period_mode=mode)
