"""
Diagnostic Script for SMC Strategy Execution
=============================================
Runs a detailed single-ticker test with debug logging to trace:
1. 1H resampling & FVG detection
2. 1H POI taps by 5M price
3. 5M micro-CHoCH signals
4. 3R Veto checks & pending limit order fills
"""

import os
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.smc.smc_structure import detect_structure_shifts, MarketStructure
from strategies.smc.smc_fvg import detect_fvgs, filter_fvgs_by_equilibrium, FVGType
from strategies.smc.smc_liquidity import find_liquidity_pools, evaluate_3r_veto, extract_pdh_pdl

# Load NVDA 5m candles
df_5m = yf.download("NVDA", interval="5m", period="60d", progress=False)
if isinstance(df_5m.columns, pd.MultiIndex):
    df_5m.columns = [c[0].lower() for c in df_5m.columns]
else:
    df_5m.columns = [c.lower() for c in df_5m.columns]

print(f"Loaded {len(df_5m)} 5m candles for NVDA.")

# 1. Test 1H resampling
df_1h = df_5m.resample("1h").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna()

print(f"Resampled to {len(df_1h)} 1H candles.")

# 2. Test 1H FVGs
fvgs_1h = detect_fvgs(df_1h, min_displacement_atr=1.0, min_body_ratio=0.55)
print(f"Detected {len(fvgs_1h)} 1H FVGs:")
for f in fvgs_1h[:5]:
    print(f"  - 1H FVG: {f.fvg_type.value} [{f.bottom:.2f}, {f.top:.2f}], mitigated={f.mitigated}, disp_ratio={f.displacement_ratio:.2f}")

# 3. Test POI taps across 5M data
poi_taps = 0
choch_count = 0
veto_passed = 0

ms_5m = MarketStructure(left_bars=2, right_bars=2)

for i in range(100, len(df_5m), 12):  # Sample every hour
    window_5m = df_5m.iloc[:i]
    cur_bar = window_5m.iloc[-1]
    cur_low = float(cur_bar["low"])
    cur_high = float(cur_bar["high"])
    cur_close = float(cur_bar["close"])

    # Check POI tap
    for f in fvgs_1h:
        if f.bar_idx * 12 > i:  # cannot use future FVG
            continue
        tapped = False
        if f.fvg_type == FVGType.BULLISH and cur_low <= f.top and cur_close >= f.bottom:
            tapped = True
        elif f.fvg_type == FVGType.BEARISH and cur_high >= f.bottom and cur_close <= f.top:
            tapped = True
        
        if tapped:
            poi_taps += 1
            # Check 5M CHoCH
            recent_5m = window_5m.iloc[-50:]
            trend, events = ms_5m.analyze(recent_5m)
            expected = "CHOCH_BULLISH" if f.fvg_type == FVGType.BULLISH else "CHOCH_BEARISH"
            has_choch = any(e.event_type == expected for e in events)
            if has_choch:
                choch_count += 1
            break

print(f"Diagnostics: POI Taps={poi_taps}, 5M CHoCH confirmations inside POI={choch_count}")

# State tracking for persistent POI
active_poi = None
poi_expiry = 0
pending_setups = 0
veto_count = 0
valid_trades = 0

for i in range(100, len(df_5m)):
    window_5m = df_5m.iloc[:i]
    cur_bar = window_5m.iloc[-1]
    cur_low = float(cur_bar["low"])
    cur_high = float(cur_bar["high"])
    cur_close = float(cur_bar["close"])

    # HTF 1H FVG detection
    df_1h_slice = df_5m.iloc[:i].resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    if len(df_1h_slice) < 10:
        continue
    
    fvgs_1h = detect_fvgs(df_1h_slice.iloc[:-1], min_displacement_atr=1.0)

    # 1. Check if price taps an unmitigated 1H FVG
    for f in fvgs_1h:
        if f.mitigated:
            continue
        if f.fvg_type == FVGType.BULLISH and cur_low <= f.top and cur_close >= f.bottom:
            active_poi = f
            poi_expiry = i + 24  # Remains active for up to 2 hours (24 5m bars)
            break
        elif f.fvg_type == FVGType.BEARISH and cur_high >= f.bottom and cur_close <= f.top:
            active_poi = f
            poi_expiry = i + 24
            break

    if active_poi and i > poi_expiry:
        active_poi = None

    if not active_poi:
        continue

    # 2. Inside active POI: Check for 5M CHoCH
    ltf_window = window_5m.iloc[-50:]
    ms = MarketStructure(left_bars=2, right_bars=2)
    trend, events = ms.analyze(ltf_window)
    expected = "CHOCH_BULLISH" if active_poi.fvg_type == FVGType.BULLISH else "CHOCH_BEARISH"
    confirmed_choch = next((e for e in reversed(events) if e.event_type == expected), None)

    if not confirmed_choch or (len(ltf_window) - 1 - confirmed_choch.bar_idx > 2):
        continue

    side = "BUY" if active_poi.fvg_type == FVGType.BULLISH else "SELL"
    origin_pivot = confirmed_choch.origin_swing
    if not origin_pivot:
        continue

    # 3. Entry: Unmitigated 5M FVG in Premium/Discount OR OTE 61.8% Retracement
    fvgs_5m = detect_fvgs(ltf_window, min_displacement_atr=0.5, min_body_ratio=0.45)
    leg_fvgs = [
        f for f in fvgs_5m
        if not f.mitigated
        and ((side == "BUY" and f.fvg_type == FVGType.BULLISH) or (side == "SELL" and f.fvg_type == FVGType.BEARISH))
    ]

    sl_price = origin_pivot.price - 0.01 if side == "BUY" else origin_pivot.price + 0.01

    if leg_fvgs:
        entry_price = leg_fvgs[-1].top if side == "BUY" else leg_fvgs[-1].bottom
    else:
        # OTE 61.8% Retracement of the confirmed CHoCH leg
        fib_diff = abs(confirmed_choch.trigger_price - origin_pivot.price)
        if side == "BUY":
            entry_price = confirmed_choch.trigger_price - 0.618 * fib_diff
        else:
            entry_price = confirmed_choch.trigger_price + 0.618 * fib_diff

    # 4. Check Liquidity Targets & 3R Veto Rule
    tp1, tp2, tp3 = find_liquidity_pools(entry_price, side, ms.pivots, fvgs_5m)
    if tp2:
        approved, rr, msg = evaluate_3r_veto(entry_price, sl_price, tp2.price, side, min_rr=3.0)
        if approved:
            pending_setups += 1
            print(f"Setup #{pending_setups} APPROVED: {side} @ ${entry_price:.2f}, SL=${sl_price:.2f}, TP2=${tp2.price:.2f} ({rr:.2f}R)")
            # reset POI to prevent duplicate entries on same reaction
            active_poi = None
        else:
            veto_count += 1

print(f"\nFinal Diagnostic Results: Approved Setups={pending_setups}, Vetoed (<3R)={veto_count}")

print(f"Results: valid_5m_fvg={valid_5m_fvg_count}, veto_count={veto_count}, pending_setups={pending_setups}")

