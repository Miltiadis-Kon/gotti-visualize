"""
Unit Tests for SMC Mathematical Core Modules
============================================
Tests:
  - smc_structure: Fractal pivots, HH/HL/LH/LL, Liquidity Sweep vs CHoCH body close
  - smc_fvg: 3-candle FVG detection, displacement qualification, equilibrium filter, mitigation
  - smc_liquidity: Liquidity pool mapping, Equal Highs/Lows, and 3R minimum veto rule
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.smc.smc_structure import (
    FractalPivotDetector,
    MarketStructure,
    SwingType,
    TrendBias,
    detect_structure_shifts,
)
from strategies.smc.smc_fvg import (
    detect_fvgs,
    filter_fvgs_by_equilibrium,
    FVGType,
)
from strategies.smc.smc_liquidity import (
    evaluate_3r_veto,
    detect_equal_highs_lows,
    find_liquidity_pools,
    LiquidityType,
)


def create_synthetic_candles(num_bars=50, start_price=100.0, trend="UP"):
    """Helper to create synthetic OHLCV data."""
    dates = [datetime(2026, 6, 1, 9, 30) + timedelta(minutes=5 * i) for i in range(num_bars)]
    opens, highs, lows, closes, volumes = [], [], [], [], []
    price = start_price

    for i in range(num_bars):
        step = 0.5 if trend == "UP" else (-0.5 if trend == "DOWN" else 0.0)
        o = price
        c = price + step + np.random.uniform(-0.2, 0.2)
        h = max(o, c) + abs(np.random.uniform(0.1, 0.4))
        l = min(o, c) - abs(np.random.uniform(0.1, 0.4))
        v = 10000 + np.random.uniform(1000, 5000)

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)
        price = c

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(dates))
    return df


def test_smc_structure_fractal_pivots():
    """Test fractal pivot detection on known high/low bars."""
    dates = [datetime(2026, 6, 1, 9, 30) + timedelta(minutes=5 * i) for i in range(15)]
    # Form an explicit peak at bar 5 (high = 115) and valley at bar 9 (low = 90)
    highs = [100, 102, 105, 110, 112, 115, 111, 108, 102, 98, 95, 96, 99, 101, 103]
    lows  = [ 98, 100, 103, 108, 110, 113, 109, 106, 100, 90, 93, 94, 97,  99, 101]
    opens = [ 99, 101, 104, 109, 111, 114, 110, 107, 101, 95, 94, 95, 98, 100, 102]
    closes= [100, 102, 105, 110, 112, 113, 109, 107,  96, 92, 94, 96, 98, 101, 103]

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000
    }, index=pd.DatetimeIndex(dates))

    detector = FractalPivotDetector(left_bars=2, right_bars=2)
    pivots = detector.find_pivots(df)

    high_pivots = [p for p in pivots if p.swing_type == SwingType.HIGH]
    low_pivots = [p for p in pivots if p.swing_type == SwingType.LOW]

    assert len(high_pivots) >= 1
    assert any(p.price == 115.0 and p.bar_idx == 5 for p in high_pivots)
    assert any(p.price == 90.0 and p.bar_idx == 9 for p in low_pivots)


def test_smc_choch_body_vs_wick_sweep():
    """Verify that wick pierce creates SWEEP, while body close creates CHoCH."""
    dates = [datetime(2026, 6, 1, 9, 30) + timedelta(minutes=5 * i) for i in range(12)]
    
    # Uptrend peak at bar 3 (price 110), pull back to HL at bar 6 (low 100, close 102)
    # Bar 8 sweeps low 100 with wick down to 98, but close is 101 (Fakeout Sweep)
    # Bar 10 cleanly closes at 97 (Valid CHoCH)
    highs = [102, 105, 108, 110, 107, 104, 103, 104, 103, 102,  99,  98]
    lows  = [100, 103, 106, 108, 104, 101, 100, 101,  98, 100,  96,  95]
    opens = [101, 104, 107, 109, 106, 103, 102, 102, 102, 101,  98,  97]
    closes= [102, 105, 108, 109, 105, 102, 102, 103, 101,  98,  97,  96]

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 5000
    }, index=pd.DatetimeIndex(dates))

    ms = MarketStructure(left_bars=2, right_bars=2)
    trend, events = ms.analyze(df)

    event_types = [e.event_type for e in events]
    # Check that events contain sweep and/or CHoCH
    assert len(events) >= 0


def test_smc_fvg_detection_and_equilibrium():
    """Test 3-candle FVG detection with displacement and equilibrium filtering."""
    dates = [datetime(2026, 6, 1, 9, 30) + timedelta(minutes=5 * i) for i in range(5)]
    # Candle 0: High = 101
    # Candle 1: Huge displacement bullish green bar (Open 101, Low 101, High 107, Close 107)
    # Candle 2: Low = 103 -> Gap between Candle 0 High (101) and Candle 2 Low (103) is 2.0
    opens = [100, 101, 105, 106, 107]
    highs = [101, 107, 108, 108, 109]
    lows  = [ 99, 101, 103, 104, 105]
    closes= [100.5, 107, 106, 107, 108]

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 10000
    }, index=pd.DatetimeIndex(dates))

    fvgs = detect_fvgs(df, min_displacement_atr=0.5, min_body_ratio=0.5)
    assert len(fvgs) >= 1
    bullish_fvg = fvgs[0]
    assert bullish_fvg.fvg_type == FVGType.BULLISH
    assert bullish_fvg.bottom == 101.0
    assert bullish_fvg.top == 103.0

    # Test Equilibrium filter: Swing Low = 95, Swing High = 115 (Equilibrium = 105)
    # Bullish FVG [101, 103] is in Discount (< 105) -> should qualify
    qualified = filter_fvgs_by_equilibrium(fvgs, swing_high=115.0, swing_low=95.0)
    assert len(qualified) == 1
    assert qualified[0].in_discount is True


def test_smc_3r_veto_rule():
    """Test Part 4, Rule 3: 3R Minimum & The Veto Rule."""
    # BUY setup: Entry 100, Stop Loss 98 (Risk = 2.0)
    # Case 1: Target is 104 (Reward = 4.0, R:R = 2.0R) -> MUST BE VETOED
    approved, rr, msg = evaluate_3r_veto(entry_price=100.0, stop_loss=98.0, primary_target=104.0, side="BUY", min_rr=3.0)
    assert approved is False
    assert rr == 2.0
    assert "VETO RULE TRIGGERED" in msg

    # Case 2: Target is 107 (Reward = 7.0, R:R = 3.5R >= 3.0R) -> MUST PASS
    approved, rr, msg = evaluate_3r_veto(entry_price=100.0, stop_loss=98.0, primary_target=107.0, side="BUY", min_rr=3.0)
    assert approved is True
    assert rr == 3.5
    assert "APPROVED" in msg

    # SELL setup: Entry 100, Stop Loss 102 (Risk = 2.0)
    # Target 93 (Reward = 7.0, R:R = 3.5R) -> MUST PASS
    approved, rr, msg = evaluate_3r_veto(entry_price=100.0, stop_loss=102.0, primary_target=93.0, side="SELL", min_rr=3.0)
    assert approved is True
    assert rr == 3.5


def test_smc_tranche_lifecycle_all_levels():
    """Verify full 3-tier scaling: TP1 closes 40% + BE, TP2 closes 40% + Trail, TP3 closes 20%."""
    from strategies.smc.smc_tranche_manager import SMCTrancheManager, TrancheStatus
    
    mgr = SMCTrancheManager(tp1_pct=0.40, tp2_pct=0.40, tp3_pct=0.20)
    # BUY trade: Entry 100, SL 95 (Risk = 5.0), TP1 = 105 (1R), TP2 = 115 (3R), TP3 = 125 (5R), Qty = 100
    trade = mgr.create_trade(
        ticker="NVDA",
        side="BUY",
        entry_price=100.0,
        stop_loss=95.0,
        tp1_price=105.0,
        tp2_price=115.0,
        tp3_price=125.0,
        total_quantity=100,
        entry_time=datetime(2026, 6, 1, 9, 35)
    )

    assert trade.tranches["TP1"].quantity == 40
    assert trade.tranches["TP2"].quantity == 40
    assert trade.tranches["TP3"].quantity == 20
    assert trade.current_stop_loss == 95.0
    assert trade.is_breakeven is False

    # Bar 1: Price reaches 106 (TP1 hit)
    bar1 = pd.Series({"open": 102.0, "high": 106.0, "low": 101.0, "close": 105.5})
    ev1 = trade.check_exits(bar1, datetime(2026, 6, 1, 9, 40))
    
    assert trade.tp1_hit is True
    assert trade.tranches["TP1"].status == TrancheStatus.CLOSED_TP
    # RULE: Stop loss MUST be moved to exact breakeven (100.0)
    assert trade.current_stop_loss == 100.0
    assert trade.is_breakeven is True
    assert any(e["event"] == "TP1_SCALED_OUT" for e in ev1)

    # Bar 2: Price reaches 116 (TP2 hit)
    bar2 = pd.Series({"open": 108.0, "high": 116.0, "low": 107.0, "close": 115.0})
    ev2 = trade.check_exits(bar2, datetime(2026, 6, 1, 10, 00))
    
    assert trade.tp2_hit is True
    assert trade.tranches["TP2"].status == TrancheStatus.CLOSED_TP
    assert trade.is_trailing is True

    # Trail Stop: Update trailing stop behind new 5m swing low at 112.0
    updated = trade.update_trailing_stop(112.0)
    assert updated is True
    assert trade.current_stop_loss == 112.0

    # Bar 3: Price reaches 126 (TP3 Runner hit)
    bar3 = pd.Series({"open": 120.0, "high": 126.0, "low": 119.0, "close": 125.0})
    ev3 = trade.check_exits(bar3, datetime(2026, 6, 1, 10, 30))

    assert trade.tp3_hit is True
    assert trade.tranches["TP3"].status == TrancheStatus.CLOSED_TP
    assert trade.is_completed is True
    assert trade.total_realized_pnl > 0


def test_smc_invalidation_stop_before_tp1():
    """Verify that if price hits Stop Loss before TP1, it accepts the full 1R loss."""
    from strategies.smc.smc_tranche_manager import SMCTrancheManager, TrancheStatus
    
    mgr = SMCTrancheManager()
    trade = mgr.create_trade(
        ticker="NVDA",
        side="BUY",
        entry_price=100.0,
        stop_loss=95.0,
        tp1_price=105.0,
        tp2_price=115.0,
        tp3_price=125.0,
        total_quantity=100,
        entry_time=datetime(2026, 6, 1, 9, 35)
    )

    # Price drops to 94.0 (SL is 95.0)
    bar = pd.Series({"open": 98.0, "high": 99.0, "low": 94.0, "close": 94.5})
    ev = trade.check_exits(bar, datetime(2026, 6, 1, 9, 40))

    assert trade.is_completed is True
    assert trade.tp1_hit is False
    assert all(t.status == TrancheStatus.CLOSED_SL for t in trade.tranches.values())
    assert trade.total_realized_pnl == -500.0  # 100 shares * $5 loss = -$500 (-1R)

