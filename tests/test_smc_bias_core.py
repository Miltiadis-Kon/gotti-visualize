"""
Unit Test Suite for 5-Rule SMC Intraday Bias Model (test_smc_bias_core.py)
===========================================================================
Validates:
1. Rule 1: M15 The Box Method (External swings, internal noise filtering, body breaks)
2. Rule 2: Session Killzones (London & NY EST time windows)
3. Rule 3: Liquidation Sweeps (Asian/Premarket sweeps as mandatory trade gates)
4. Rule 4: M1 Micro-CHoCH (Impulsive candle body close confirmation)
5. Rule 5: M5 POI Selection (Order Block, FVG, IFVG & Confluence Stacking)
6. Asymmetric 1:5R Veto Rule & Hybrid Partial Scaling Lifecycle
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from strategies.smc_bias_model.bias_structure import M15BoxStructureDetector, BiasDirection, SwingPointType
from strategies.smc_bias_model.session_killzones import SessionKillzoneManager, SessionExtremes
from strategies.smc_bias_model.m1_choch import M1CHoCHDetector
from strategies.smc_bias_model.m5_poi import M5POISelector, POIType
from strategies.smc_bias_model.bias_order_manager import BiasOrderManager, OrderLifecycleStatus


def test_m15_box_method_and_body_breaks():
    """Validates Rule 1: The Box Method ignores internal noise and requires candle body closes."""
    detector = M15BoxStructureDetector(left_bars=2, right_bars=2)
    dates = pd.date_range("2026-09-01 09:30:00", periods=20, freq="15min", tz="UTC")

    # Construct clean downtrend: High 105 -> Low 95 -> Retracement to 100 -> Break to 90
    opens =  [100, 102, 104, 101,  98,  96,  95,  97,  99, 100,  98,  96,  94,  92,  91,  89,  90,  90,  90,  90]
    highs =  [102, 104, 106, 102,  99,  97,  96,  98, 100, 101,  99,  97,  95,  93,  92,  90,  91,  91,  91,  91]
    lows =   [ 99, 101, 103,  97,  95,  94,  94,  96,  97,  98,  96,  93,  91,  90,  89,  88,  89,  89,  89,  89]
    closes = [102, 104, 101,  98,  96,  95,  96,  97,  99,  98,  96,  94,  92,  91,  89,  89,  90,  90,  90,  90]

    df_m15 = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000
    }, index=dates)

    bias, box = detector.analyze_external_structure(df_m15)
    assert bias == BiasDirection.BEARISH
    assert box is not None
    assert box.bias == BiasDirection.BEARISH
    assert box.target_price < box.invalidation_price


def test_session_killzones_and_liquidation_sweeps():
    """Validates Rules 2 & 3: Killzone time verification and Asian/Premarket sweeps."""
    mgr = SessionKillzoneManager(allow_ny_extension=True)

    # 1. Test Killzones (EST)
    tz_ny = pytz.timezone("America/New_York")
    london_time = tz_ny.localize(datetime(2026, 9, 15, 3, 30))  # 03:30 AM EST
    ny_time = tz_ny.localize(datetime(2026, 9, 15, 9, 45))      # 09:45 AM EST
    afternoon_time = tz_ny.localize(datetime(2026, 9, 15, 13, 0)) # 01:00 PM EST

    in_london, s_london = mgr.is_in_killzone(london_time)
    assert in_london is True
    assert s_london == "LONDON_KILLZONE"

    in_ny, s_ny = mgr.is_in_killzone(ny_time)
    assert in_ny is True
    assert s_ny == "NEW_YORK_KILLZONE"

    in_afternoon, _ = mgr.is_in_killzone(afternoon_time)
    assert in_afternoon is False

    # 2. Test Liquidation Sweep
    benchmarks = {
        "ASIAN": SessionExtremes(high=205.0, low=200.0, high_time=None, low_time=None, session_name="ASIAN_SESSION"),
        "PREMARKET": SessionExtremes(high=206.0, low=201.0, high_time=None, low_time=None, session_name="PREMARKET"),
    }

    # Short scenario: price wicks above 206.0 -> valid sweep
    sweep_short = mgr.check_liquidation_sweep(
        cur_high=206.50, cur_low=204.0, bias="BEARISH", active_killzone="NEW_YORK_KILLZONE",
        benchmarks=benchmarks, cur_time=ny_time
    )
    assert sweep_short.swept is True
    assert sweep_short.sweep_type == "HIGH_SWEPT"

    # Short scenario without sweeping high -> no sweep
    no_sweep = mgr.check_liquidation_sweep(
        cur_high=204.50, cur_low=203.0, bias="BEARISH", active_killzone="NEW_YORK_KILLZONE",
        benchmarks=benchmarks, cur_time=ny_time
    )
    assert no_sweep.swept is False


def test_m1_choch_body_close_vs_wick_sweep():
    """Validates Rule 4: 1-minute CHoCH triggers on solid candle body close."""
    detector = M1CHoCHDetector(left_bars=2, right_bars=2)
    dates = pd.date_range("2026-09-15 09:30:00", periods=15, freq="1min", tz="UTC")

    # In a bearish bias, counter-trend pullback forms HL at 102.0. Then price breaks below 102.0.
    # Bar 5 forms low at 102.0. Bar 8 forms high at 105.0. Bar 12 closes at 101.50 (body close break).
    opens =  [100.0, 101.0, 102.0, 103.0, 102.5, 102.0, 103.0, 104.0, 104.5, 104.0, 103.0, 102.5, 102.2, 101.8, 101.5]
    highs =  [101.5, 102.5, 103.5, 103.5, 103.0, 103.0, 104.5, 105.5, 105.0, 104.5, 103.5, 102.8, 102.3, 102.0, 101.6]
    lows =   [ 99.8, 100.8, 101.8, 102.2, 101.9, 101.9, 102.8, 103.5, 103.8, 102.8, 102.2, 101.9, 101.4, 101.2, 101.0]
    closes = [101.0, 102.0, 103.0, 102.5, 102.1, 102.8, 104.0, 105.0, 104.0, 103.0, 102.5, 102.1, 101.6, 101.4, 101.1]

    df_m1 = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 500}, index=dates)

    choch = detector.detect_m1_choch(df_m1, bias="BEARISH")
    assert choch is not None
    assert choch.event_type == "M1_CHOCH_BEARISH"
    assert choch.structural_extreme_price >= 105.0


def test_m5_poi_ob_fvg_confluence():
    """Validates Rule 5: 5-minute Order Block, FVG, and confluence stacking."""
    selector = M5POISelector(min_displacement_ratio=1.0)
    dates = pd.date_range("2026-09-15 09:30:00", periods=8, freq="5min", tz="UTC")

    # Create Bearish OB (green candle at bar 2) followed by aggressive drop leaving an FVG
    opens =  [100.0, 101.0, 102.0, 104.0, 100.0,  97.0,  95.0,  94.0]
    highs =  [101.5, 102.5, 104.5, 105.0, 100.5,  97.5,  96.0,  95.0]
    lows =   [ 99.5, 100.5, 101.8, 100.0,  96.5,  94.5,  93.5,  93.0]
    closes = [101.0, 102.0, 104.0, 100.2,  97.0,  95.0,  94.0,  93.5]

    df_m5 = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 2000}, index=dates)

    poi = selector.select_best_poi_with_confluence(df_m5, bias="BEARISH")
    assert poi is not None
    assert poi.bias == "BEARISH"
    assert poi.proximate_price > 0


def test_asymmetric_5r_veto_rule():
    """Validates the strict 1:5.0R minimum reward-to-risk veto rule."""
    order_mgr = BiasOrderManager(min_rr=5.0)

    # Valid Setup: Entry=100.0, SL=102.0 (Risk=2.0), Target=88.0 (Reward=12.0) -> R:R = 6.0R (>= 5.0R)
    approved, rr, msg = order_mgr.evaluate_5r_veto(entry_price=100.0, stop_loss=102.0, m15_target=88.0, side="SELL")
    assert approved is True
    assert rr == 6.0

    # Invalid Setup: Entry=100.0, SL=102.0 (Risk=2.0), Target=93.0 (Reward=7.0) -> R:R = 3.5R (< 5.0R)
    rejected, rr_bad, msg_bad = order_mgr.evaluate_5r_veto(entry_price=100.0, stop_loss=102.0, m15_target=93.0, side="SELL")
    assert rejected is False
    assert rr_bad == 3.5
    assert "VETO" in msg_bad


def test_hybrid_partial_scaling_lifecycle():
    """Validates the user's hybrid scaling: 40% TP1 @ 3.0R (+ breakeven) and 60% runner @ M15 target."""
    order_mgr = BiasOrderManager(min_rr=5.0, risk_pct=0.01, tp1_pct=0.40, tp2_pct=0.60)
    trade = order_mgr.create_trade(
        ticker="NVDA",
        side="SELL",
        entry_price=100.0,
        structural_extreme_price=102.0,
        m15_target_price=88.0,
        equity=100_000.0,
        entry_time=datetime(2026, 9, 15, 9, 35)
    )

    assert trade is not None
    assert trade.risk_per_share == pytest.approx(2.01, abs=0.01)
    assert trade.tp1_price == pytest.approx(100.0 - 3.0 * trade.risk_per_share, abs=0.05)
    assert trade.tp2_target_price == 88.0

    # 1. Bar reaches TP1 -> 40% scaled out, SL moved to breakeven
    bar_tp1 = pd.Series({"open": 96.0, "high": 96.5, "low": 93.0, "close": 93.5})
    events1 = trade.check_exits(bar_tp1, datetime(2026, 9, 15, 9, 50))

    assert trade.tp1_hit is True
    assert trade.current_stop_loss == 100.0  # Moved to breakeven!
    assert trade.total_realized_pnl > 0

    # 2. Bar reaches M15 External Target (88.0) -> 60% runner scaled out
    bar_tp2 = pd.Series({"open": 90.0, "high": 90.5, "low": 87.5, "close": 88.0})
    events2 = trade.check_exits(bar_tp2, datetime(2026, 9, 15, 10, 15))

    assert trade.tp2_hit is True
    assert trade.is_completed is True
    assert trade.status == OrderLifecycleStatus.COMPLETED_WIN
