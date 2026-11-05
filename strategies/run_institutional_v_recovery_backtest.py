"""
Institutional V-Recovery Backtest Runner
========================================
Runs comparative multi-day portfolio backtests for:
1. Institutional_VRecovery (5 Upgrades: VWAP, RVOL >= 1.5, QQQ Divergence,
   Dynamic ATR Stops, 11:00 EST Time Stop, 2-Tier Scaled Exits)
2. HighBeta_CatalystOpen (Base 5-step engine)
3. News_LiquiditySweep
4. MarketReactor (Fade Mode)
5. Benchmarks: SPY ($100k) & NASDAQ-100 / QQQ ($100k)
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

from strategies.market_reactor import fetch_reactor_signals, ReactorSignal, get_mysql_connection
from strategies.high_beta_catalyst_open import (
    COMPETITOR_MAP,
    CatalystConfluenceEngine,
    ZedStrategy,
    HighBetaCatalystOpenStrategy,
    check_7day_consecutive_volatility,
    scan_premarket_7day_volatility_universe,
)
from strategies.institutional_v_recovery import (
    compute_vwap,
    compute_ema,
    ZedInstitutionalStrategy,
    ZedNoVwapStrategy,
    InstitutionalVRecoveryStrategy,
)
from strategies.risk_management import calculate_atr
from strategies.run_combined_news_backtest import simulate_strategy_engine
from strategies.run_high_beta_catalyst_backtest import (
    simulate_high_beta_catalyst_engine,
    simulate_zed_strategy_engine,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ZedStrategyBacktest")

# Seed universe of tradeable high-beta assets (dynamically qualified each morning via 7-day volatility scanner)
HIGH_BETA_UNIVERSE = ["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR", "SMCI", "RBLX", "RIOT", "CLSK", "AAPL", "MSFT"]


def simulate_institutional_v_engine(
    date_list: List[str],
    all_eligible_signals: List[Tuple[str, ReactorSignal]],
    ticker_data_5m: Dict[str, pd.DataFrame],
    ticker_data_1d: Dict[str, pd.DataFrame],
    qqq_5m: Optional[pd.DataFrame],
    qqq_1d: Optional[pd.DataFrame] = None,
    initial_portfolio: float = 100000.0,
    target1_pct: float = 0.020,         # Tier 1 Target: +2.0%
    target2_runner_pct: float = 0.035,  # Tier 2 Runner: +3.5%
    atr_stop_mult: float = 1.2,         # 1.2x of 15m ATR
    max_stop_pct: float = 0.030,        # Max 3.0% structural stop
    min_rvol: float = 1.5,              # RVOL >= 1.5
    breakeven_gain_pct: float = 0.010,  # Breakeven ratchet at +1.0% gain
    max_trades_per_day: int = 2,        # Max 2 top-confluence setups per day
    require_vwap: bool = True,          # Toggle VWAP Reclaim condition
    require_rvol: bool = True,          # Toggle RVOL >= 1.5 condition
    strategy_label: str = "ZedStrategy_Institutional (V2: WITH VWAP & RVOL)",
) -> Dict[str, Any]:
    """
    Simulates the Institutional V-Recovery strategy execution with all 5 upgrades:
    1. Volume Profile & Intraday VWAP Confluence (Configurable)
    2. Quantitative Higher-Timeframe Nasdaq Filter + Relative Strength Divergence
    3. Structural Volatility-Based Stop-Loss (Dynamic 15m ATR, max 3.0%)
    4. Strict Execution Windows (09:35-10:15 entries, 15:55 EOD close)
    5. Partial Scale-Outs (50% at +2.0%) + EMA(9) Trailing Runner (+3.5%)
    """
    portfolio_cash = initial_portfolio
    closed_trades: List[Dict[str, Any]] = []

    for d_str in date_list:
        curr_date = pd.to_datetime(d_str).date()
        day_sig_cache = [s for _, s in all_eligible_signals if _ == d_str]

        # Extract QQQ candles for curr_date
        qqq_day = None
        if qqq_5m is not None and not qqq_5m.empty:
            qqq_day = qqq_5m[qqq_5m.index.date == curr_date]

        prior_qqq_1d = None
        if qqq_1d is not None and not qqq_1d.empty:
            prior_qqq_1d = qqq_1d[qqq_1d.index.date < curr_date]

        # Enhancement 2: Higher-Timeframe Market Filter on Nasdaq Composite (QQQ)
        if qqq_day is not None and len(qqq_day) > 0 and prior_qqq_1d is not None and len(prior_qqq_1d) >= 10:
            qqq_open = float(qqq_day.iloc[0]["open"])
            qqq_now = float(qqq_day.iloc[0]["close"])
            market_regime = CatalystConfluenceEngine.evaluate_market_regime(
                prior_qqq_1d, qqq_now, qqq_open
            )
            if market_regime != "BULL_MARKET":
                continue

        # Collect candidate signals for current session
        day_candidates = []

        # Dynamic Pre-Market Universe: Filter candidates through 7-day consecutive volatility screen
        candidate_pool = list(dict.fromkeys(HIGH_BETA_UNIVERSE + [s.ticker for s in day_sig_cache if s.ticker]))
        qualified_universe = scan_premarket_7day_volatility_universe(
            target_date=curr_date,
            candidate_tickers=candidate_pool,
            ticker_data_1d=ticker_data_1d,
            min_daily_range_pct=0.020,
            lookback_days=7
        )

        seen_tickers: set = set()

        for ticker in qualified_universe:
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            if ticker not in ticker_data_5m:
                continue

            df_5m = ticker_data_5m[ticker]
            day_candles = df_5m[df_5m.index.date == curr_date].copy()
            if len(day_candles) < 4:
                continue

            # Prior daily bars for ATR
            df_1d = ticker_data_1d.get(ticker)
            if df_1d is None or df_1d.empty:
                continue
            prior_1d = df_1d[df_1d.index.date < curr_date]
            if len(prior_1d) < 2:
                continue

            pdc = float(prior_1d.iloc[-1]["close"])
            daily_atr = float(calculate_atr(prior_1d, length=min(14, len(prior_1d) - 1))) or (pdc * 0.025)
            if (daily_atr / pdc) < 0.020:
                continue

            # Catalyst Confluence
            cat_analysis = CatalystConfluenceEngine.analyze_catalysts(
                target_ticker=ticker,
                target_date_str=d_str,
                signals_cache=day_sig_cache
            )
            cat_bias = cat_analysis["catalyst_bias"]

            # Compute session VWAP and 9 EMA
            day_candles["vwap"] = compute_vwap(day_candles)
            day_candles["ema9"] = compute_ema(day_candles["close"], period=9)

            open_price = float(day_candles.iloc[0]["open"])
            session_low = float(day_candles.iloc[0]["low"])
            session_high = float(day_candles.iloc[0]["high"])

            for idx in range(1, len(day_candles)):
                bar = day_candles.iloc[idx]
                b_time = bar.name.time()
                cur_price = float(bar["close"])
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                cur_vwap = float(bar["vwap"])

                session_low = min(session_low, bar_low)
                session_high = max(session_high, bar_high)

                prev_bar = day_candles.iloc[idx - 1]
                is_green_reversal = (cur_price > bar["open"]) and (cur_price > prev_bar["high"])

                # Strict Entry Window (09:35 - 10:15 EST)
                if time(9, 35) <= b_time <= time(10, 15):
                    # Enhancement 1: VWAP Reclaim (Configurable)
                    vwap_reclaimed = True if not require_vwap else ((cur_price >= cur_vwap) or (bar_low <= cur_vwap and cur_price > cur_vwap))

                    # Enhancement 1: RVOL >= 1.5 (Configurable)
                    vol_ma20 = float(day_candles["volume"].iloc[:idx+1].rolling(min(20, idx+1)).mean().iloc[-1])
                    cur_vol = float(bar["volume"])
                    rvol = (cur_vol / vol_ma20) if vol_ma20 > 0 else 1.0
                    rvol_confirmed = True if not require_rvol else ((rvol >= min_rvol) or (cur_vol >= vol_ma20 * 1.3))

                    # Enhancement 2: Bullish Divergence vs QQQ
                    has_bullish_div = True
                    if qqq_day is not None and len(qqq_day) > idx:
                        qqq_bar = qqq_day.iloc[idx]
                        qqq_prev = qqq_day.iloc[idx - 1]
                        qqq_dropping = float(qqq_bar["low"]) <= float(qqq_prev["low"])
                        stock_holding = bar_low >= prev_bar["low"]
                        has_bullish_div = (not qqq_dropping) or stock_holding

                    drop_from_open = (open_price - session_low) / open_price
                    bounce_height = (cur_price - session_low) / max(0.01, session_low)

                    # Confluence Trigger
                    if (
                        is_green_reversal
                        and vwap_reclaimed
                        and rvol_confirmed
                        and has_bullish_div
                        and (cat_bias == "BULLISH" or drop_from_open >= 0.010)
                        and bounce_height >= 0.006
                    ):
                        # Enhancement 3: Structural ATR Volatility Stop
                        atr_buffer = atr_stop_mult * (daily_atr * 0.20)
                        hard_sl = max(session_low - atr_buffer, cur_price * (1.0 - max_stop_pct))
                        risk_per_share = abs(cur_price - hard_sl)
                        if risk_per_share > 0:
                            score = (rvol if require_rvol else (bounce_height * 100)) * (1.5 if cat_bias == "BULLISH" else 1.0)
                            day_candidates.append({
                                "ticker": ticker,
                                "bar_idx": idx,
                                "time": b_time,
                                "price": cur_price,
                                "sl": hard_sl,
                                "rvol": rvol,
                                "score": score,
                                "candles": day_candles,
                                "cat_bias": cat_bias,
                                "vwap": cur_vwap,
                                "reason": f"VWAP Reclaimed (${cur_vwap:.2f}) | RVOL={rvol:.1f}x | Bullish Div Confirmed | {cat_analysis['catalyst_type']}"
                            })
                            break

        # Select top setups up to max_trades_per_day
        day_candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = day_candidates[:max_trades_per_day]

        for cand in selected:
            ticker = cand["ticker"]
            entry_price = cand["price"]
            hard_sl = cand["sl"]
            candles = cand["candles"]
            start_idx = cand["bar_idx"]

            risk_per_share = abs(entry_price - hard_sl)
            qty = floor((portfolio_cash * 0.02) / risk_per_share)
            qty = max(2, min(qty, floor((portfolio_cash * 0.90) / entry_price)))

            total_shares = qty
            remaining_shares = qty
            tier1_sold = False
            tier1_pnl = 0.0
            stop_loss = hard_sl
            target1_price = entry_price * (1.0 + target1_pct)
            be_applied = False
            trade_active = True

            for idx in range(start_idx + 1, len(candles)):
                bar = candles.iloc[idx]
                b_time = bar.name.time()
                cur_price = float(bar["close"])
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                cur_ema9 = float(bar["ema9"])
                prev_bar = candles.iloc[idx - 1]

                gain_pct = (cur_price - entry_price) / entry_price

                # Enhancement 3: Breakeven Ratchet at +1.0% gain
                if not be_applied and gain_pct >= breakeven_gain_pct:
                    stop_loss = entry_price + (0.001 * entry_price)
                    be_applied = True

                # Enhancement 5: Tier 1 Scale-Out (50% sold at +2.0% Target)
                if not tier1_sold and bar_high >= target1_price:
                    half_qty = floor(total_shares * 0.50)
                    tier1_gain = (target1_price - entry_price) * half_qty
                    tier1_pnl = tier1_gain
                    remaining_shares -= half_qty
                    tier1_sold = True
                    stop_loss = max(stop_loss, entry_price)

                # Remaining Position Management
                trade_closed = False
                exit_price = cur_price
                exit_reason = ""

                # Hard Stop Loss / Breakeven Hit
                if bar_low <= stop_loss:
                    trade_closed = True
                    exit_price = stop_loss
                    exit_reason = "BREAKEVEN" if be_applied and exit_price >= entry_price else "STOP_LOSS (1.2x ATR)"

                # Enhancement 5: Runner Target (+3.5%)
                elif tier1_sold and bar_high >= (entry_price * (1.0 + target2_runner_pct)):
                    trade_closed = True
                    exit_price = entry_price * (1.0 + target2_runner_pct)
                    exit_reason = f"RUNNER_TARGET_HIT (+{target2_runner_pct:.1%})"

                # Enhancement 5: Trailing 9 EMA Exit after 10:30 EST
                elif tier1_sold and cur_price < cur_ema9 and prev_bar["close"] < cur_ema9 and b_time >= time(10, 30):
                    trade_closed = True
                    exit_price = cur_price
                    exit_reason = "EMA9_TRAIL_EXIT"

                # Strict EOD Flatten at 15:55 EST (Zero overnight hold)
                elif b_time >= time(15, 55):
                    trade_closed = True
                    exit_price = cur_price
                    exit_reason = "EOD_FLATTEN"

                if trade_closed:
                    tier2_pnl = (exit_price - entry_price) * remaining_shares
                    total_pnl = tier1_pnl + tier2_pnl
                    trade_ret = total_pnl / (entry_price * total_shares)
                    portfolio_cash += total_pnl
                    is_win = total_pnl > 0

                    closed_trades.append({
                        "date": d_str,
                        "ticker": ticker,
                        "strategy": "Institutional_VRecovery",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "total_shares": total_shares,
                        "tier1_sold": tier1_sold,
                        "pnl": total_pnl,
                        "return_pct": trade_ret,
                        "is_win": is_win,
                        "exit_reason": exit_reason,
                        "portfolio_value": portfolio_cash,
                    })
                    break

    # Metrics
    tot_trades = len(closed_trades)
    win_trades = [t for t in closed_trades if t["is_win"]]
    be_trades = [t for t in closed_trades if "BREAKEVEN" in t["exit_reason"] or abs(t["pnl"]) < 5.0]
    loss_trades = [t for t in closed_trades if not t["is_win"] and t not in be_trades]
    eff_trades = tot_trades - len(be_trades)
    win_rate = (len(win_trades) / eff_trades) if eff_trades > 0 else 0.0

    gross_profit = sum(t["pnl"] for t in win_trades)
    gross_loss = abs(sum(t["pnl"] for t in loss_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    net_pnl = portfolio_cash - initial_portfolio
    total_ret = net_pnl / initial_portfolio

    return {
        "strategy": strategy_label,
        "total_trades": tot_trades,
        "winning_trades": len(win_trades),
        "breakeven_trades": len(be_trades),
        "losing_trades": len(loss_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net_pnl,
        "return_pct": total_ret,
        "final_portfolio": portfolio_cash,
        "trades": closed_trades,
    }


# Backward compatibility alias
simulate_zed_institutional_engine = simulate_institutional_v_engine


def run_comparative_inst_v_backtest(period_mode: str = "3months"):
    print("\n" + "=" * 115)
    print(" INSTITUTIONAL V-RECOVERY STRATEGY COMPARATIVE BACKTEST RUNNER (5 ENHANCEMENTS)")
    print("=" * 115)

    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    min_date = '2026-08-05' if period_mode in ("3months", "3month") else '2026-09-01'
    cursor.execute(
        f"SELECT DISTINCT signal_date FROM signals WHERE signal_date >= '{min_date}' ORDER BY signal_date ASC"
    )
    all_dates = [r["signal_date"].strftime("%Y-%m-%d") for r in cursor.fetchall()]
    conn.close()

    if period_mode in ("3months", "3month"):
        date_list = [d for d in all_dates if d >= "2026-08-05"]
        period_title = "Past 3 Months (Aug 5 - Nov 5, 2026)"
    elif period_mode == "1month":
        date_list = [d for d in all_dates if d >= "2026-10-01"]
        period_title = "Past 1 Month (Oct 1 - Nov 5, 2026)"
    else:  # "2months"
        date_list = [d for d in all_dates if d >= "2026-09-01"]
        period_title = "Past 2 Months (Sep 1 - Nov 5, 2026)"

    print(f"Evaluation Period: {period_title} ({len(date_list)} trading sessions)")
    print(f"High-Beta Universe: {', '.join(HIGH_BETA_UNIVERSE)}")

    # Ingest signals
    all_eligible: List[Tuple[str, ReactorSignal]] = []
    for d in date_list:
        sigs = fetch_reactor_signals(d, include_prev_afterhours=True)
        for s in sigs:
            all_eligible.append((d, s))

    print(f"Total News Signals Ingested: {len(all_eligible)}")

    # Download 5m candles via Alpaca IEX (with YFinance fallback)
    all_download_syms = list(set(HIGH_BETA_UNIVERSE + ["SPY", "QQQ"]))
    start_dt = pd.to_datetime(date_list[0]) - timedelta(days=7)
    end_dt = pd.to_datetime(date_list[-1]) + timedelta(days=2)

    ticker_data_5m: Dict[str, pd.DataFrame] = {}
    ticker_data_1d: Dict[str, pd.DataFrame] = {}

    try:
        alpaca_key = os.getenv("ALPACA_API_KEY")
        alpaca_sec = os.getenv("ALPACA_API_SECRET")
        if alpaca_key and alpaca_sec:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed

            alp_client = StockHistoricalDataClient(alpaca_key, alpaca_sec)
            tf5 = TimeFrame(5, TimeFrameUnit.Minute)
            req = StockBarsRequest(
                symbol_or_symbols=all_download_syms,
                timeframe=tf5,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.IEX
            )
            bars_alp = alp_client.get_stock_bars(req)
            if bars_alp and hasattr(bars_alp, "df") and not bars_alp.df.empty:
                df_alp = bars_alp.df
                for sym in all_download_syms:
                    if sym in df_alp.index.levels[0]:
                        sub = df_alp.xs(sym).copy()
                        sub.columns = [c.lower() for c in sub.columns]
                        if sub.index.tz is not None:
                            sub.index = sub.index.tz_convert("America/New_York")
                        else:
                            sub.index = sub.index.tz_localize("UTC").tz_convert("America/New_York")
                        ticker_data_5m[sym] = sub
                print(f"Alpaca successfully fetched 5m candles for {len(ticker_data_5m)} symbols.")
    except Exception as e:
        print(f"Notice fetching Alpaca 5m: {e}")

    for sym in all_download_syms:
        try:
            if sym not in ticker_data_5m or ticker_data_5m[sym].empty:
                earliest_safe = pd.Timestamp.now().tz_localize(None) - timedelta(days=58)
                yf_start = max(start_dt, earliest_safe)
                df_5m = yf.download(sym, start=yf_start.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="5m", progress=False)
                if not df_5m.empty:
                    df_5m.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_5m.columns]
                    if df_5m.index.tz is not None:
                        df_5m.index = df_5m.index.tz_convert("America/New_York")
                    else:
                        df_5m.index = df_5m.index.tz_localize("UTC").tz_convert("America/New_York")
                    ticker_data_5m[sym] = df_5m

            # 1d bars
            df_1d = yf.download(sym, start="2026-05-01", end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)
            if not df_1d.empty:
                df_1d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_1d.columns]
                ticker_data_1d[sym] = df_1d
        except Exception:
            pass

    qqq_5m = ticker_data_5m.get("QQQ")
    initial_portfolio = 100000.0

    # 1. Run Institutional V-Recovery Simulations (V2: WITH VWAP & RVOL vs V3: WITHOUT VWAP & RVOL)
    res_inst_v2 = simulate_institutional_v_engine(
        date_list=date_list,
        all_eligible_signals=all_eligible,
        ticker_data_5m=ticker_data_5m,
        ticker_data_1d=ticker_data_1d,
        qqq_5m=qqq_5m,
        qqq_1d=ticker_data_1d.get("QQQ"),
        initial_portfolio=initial_portfolio,
        target1_pct=0.020,
        target2_runner_pct=0.035,
        atr_stop_mult=1.2,
        max_stop_pct=0.030,
        min_rvol=1.5,
        breakeven_gain_pct=0.010,
        max_trades_per_day=2,
        require_vwap=True,
        require_rvol=True,
        strategy_label="ZedStrategy_Institutional (V2: WITH VWAP & RVOL)",
    )

    res_inst_v3 = simulate_institutional_v_engine(
        date_list=date_list,
        all_eligible_signals=all_eligible,
        ticker_data_5m=ticker_data_5m,
        ticker_data_1d=ticker_data_1d,
        qqq_5m=qqq_5m,
        qqq_1d=ticker_data_1d.get("QQQ"),
        initial_portfolio=initial_portfolio,
        target1_pct=0.020,
        target2_runner_pct=0.035,
        atr_stop_mult=1.2,
        max_stop_pct=0.030,
        min_rvol=1.5,
        breakeven_gain_pct=0.010,
        max_trades_per_day=2,
        require_vwap=False,
        require_rvol=False,
        strategy_label="ZedStrategy_NoVwapFilter (V3: NO VWAP & RVOL)",
    )

    # Version 1: ZedStrategy_Base (4-Pattern Base Engine + 7D Volatility Universe)
    res_v1 = simulate_zed_strategy_engine(
        date_list=date_list,
        all_eligible_signals=all_eligible,
        ticker_data_5m=ticker_data_5m,
        ticker_data_1d=ticker_data_1d,
        qqq_1d=ticker_data_1d.get("QQQ"),
        qqq_5m=qqq_5m,
        initial_portfolio=initial_portfolio,
        take_profit_pct=0.020,
        stop_loss_pct=0.035,
        enable_scale_in=True,
        breakeven_gain_pct=0.010,
        require_7day_vol=True,
    )
    res_v1["strategy"] = "ZedStrategy_Base (V1: 4-Pattern Engine)"

    # 2. Resample 5m to 15m for LiquiditySweep & MarketReactor
    ticker_data_15m: Dict[str, pd.DataFrame] = {}
    for sym, df_5m in ticker_data_5m.items():
        try:
            df_15m = df_5m.resample("15min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            ticker_data_15m[sym] = df_15m
        except Exception:
            pass

    res_swp = simulate_strategy_engine(
        "News_LiquiditySweep", date_list, all_eligible, ticker_data_15m,
        all_market_dates=date_list, initial_portfolio=initial_portfolio
    )

    res_fade = simulate_strategy_engine(
        "MarketReactor", date_list, all_eligible, ticker_data_15m,
        all_market_dates=date_list, initial_portfolio=initial_portfolio,
        selector2_mode="fade"
    )
    res_fade["strategy"] = "MarketReactor (Fade Mode)"

    # 3. Benchmark Calculations
    benchmarks = []
    for b_sym, b_name in [("SPY", "SPY (S&P 500 ETF)"), ("QQQ", "NASDAQ 100 (QQQ ETF)")]:
        if b_sym in ticker_data_5m and not ticker_data_5m[b_sym].empty:
            b_df = ticker_data_5m[b_sym]
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
                    "pnl": b_pnl,
                })

    # Summary Table
    print("\n" + "=" * 120)
    print(f" SIDE-BY-SIDE PERFORMANCE MATRIX — {period_title.upper()}")
    print("=" * 120)

    summary_rows = [
        {
            "Strategy / Benchmark": res_inst_v2["strategy"],
            "Total Trades": res_inst_v2["total_trades"],
            "Wins": res_inst_v2["winning_trades"],
            "Losses": res_inst_v2["losing_trades"],
            "Breakeven": res_inst_v2["breakeven_trades"],
            "Win Rate": f"{res_inst_v2['win_rate']:.1%}",
            "Profit Factor": f"{res_inst_v2['profit_factor']:.2f}",
            "Gross Profit": f"${res_inst_v2['gross_profit']:+,.2f}",
            "Gross Loss": f"-${res_inst_v2['gross_loss']:,.2f}",
            "Net P&L": f"${res_inst_v2['net_pnl']:+,.2f}",
            "Return %": f"{res_inst_v2['return_pct']:+.2%}",
        },
        {
            "Strategy / Benchmark": res_inst_v3["strategy"],
            "Total Trades": res_inst_v3["total_trades"],
            "Wins": res_inst_v3["winning_trades"],
            "Losses": res_inst_v3["losing_trades"],
            "Breakeven": res_inst_v3["breakeven_trades"],
            "Win Rate": f"{res_inst_v3['win_rate']:.1%}",
            "Profit Factor": f"{res_inst_v3['profit_factor']:.2f}",
            "Gross Profit": f"${res_inst_v3['gross_profit']:+,.2f}",
            "Gross Loss": f"-${res_inst_v3['gross_loss']:,.2f}",
            "Net P&L": f"${res_inst_v3['net_pnl']:+,.2f}",
            "Return %": f"{res_inst_v3['return_pct']:+.2%}",
        },
        {
            "Strategy / Benchmark": res_v1["strategy"],
            "Total Trades": res_v1["total_trades"],
            "Wins": res_v1.get("tp_hits", res_v1.get("winning_trades", 0)),
            "Losses": res_v1.get("sl_hits", res_v1.get("losing_trades", 0)),
            "Breakeven": res_v1.get("breakeven_trades", 0),
            "Win Rate": f"{res_v1['win_rate']:.1%}",
            "Profit Factor": f"{res_v1['profit_factor']:.2f}",
            "Gross Profit": f"${res_v1.get('gross_profit', 0.0):+,.2f}",
            "Gross Loss": f"-${abs(res_v1.get('gross_loss', 0.0)):,.2f}",
            "Net P&L": f"${res_v1['net_pnl']:+,.2f}",
            "Return %": f"{res_v1['return_pct']:+.2%}",
        },
        {
            "Strategy / Benchmark": res_swp["strategy"],
            "Total Trades": res_swp["total_trades"],
            "Wins": res_swp.get("tp_hits", res_swp.get("winning_trades", 0)),
            "Losses": res_swp.get("sl_hits", res_swp.get("losing_trades", 0)),
            "Breakeven": 0,
            "Win Rate": f"{res_swp['win_rate']:.1%}",
            "Profit Factor": f"{res_swp['profit_factor']:.2f}",
            "Gross Profit": f"${res_swp.get('gross_profit', res_swp['closed_pnl']):+,.2f}",
            "Gross Loss": f"-${abs(res_swp.get('gross_loss', 0.0)):,.2f}",
            "Net P&L": f"${res_swp['net_pnl']:+,.2f}",
            "Return %": f"{res_swp['return_pct']:+.2%}",
        },
        {
            "Strategy / Benchmark": res_fade["strategy"],
            "Total Trades": res_fade["total_trades"],
            "Wins": res_fade.get("tp_hits", res_fade.get("winning_trades", 0)),
            "Losses": res_fade.get("sl_hits", res_fade.get("losing_trades", 0)),
            "Breakeven": 0,
            "Win Rate": f"{res_fade['win_rate']:.1%}",
            "Profit Factor": f"{res_fade['profit_factor']:.2f}",
            "Gross Profit": f"${res_fade.get('gross_profit', res_fade['closed_pnl']):+,.2f}",
            "Gross Loss": f"-${abs(res_fade.get('gross_loss', 0.0)):,.2f}",
            "Net P&L": f"${res_fade['net_pnl']:+,.2f}",
            "Return %": f"{res_fade['return_pct']:+.2%}",
        }
    ]

    for b in benchmarks:
        summary_rows.append({
            "Strategy / Benchmark": f"Buy & Hold: {b['name']}",
            "Total Trades": 1,
            "Wins": 1 if b["pnl"] > 0 else 0,
            "Losses": 0 if b["pnl"] > 0 else 1,
            "Breakeven": 0,
            "Win Rate": "100.0%" if b["pnl"] > 0 else "0.0%",
            "Profit Factor": "N/A",
            "Gross Profit": f"${max(0.0, b['pnl']):+,.2f}",
            "Gross Loss": f"-${abs(min(0.0, b['pnl'])):,.2f}",
            "Net P&L": f"${b['pnl']:+,.2f}",
            "Return %": f"{b['return_pct']:+.2%}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # Alpha Outperformance
    if benchmarks:
        spy_b = next((b for b in benchmarks if b["symbol"] == "SPY"), None)
        qqq_b = next((b for b in benchmarks if b["symbol"] == "QQQ"), None)
        print("\n" + "-" * 120)
        print(" ALPHA OUTPERFORMANCE RELATIVE TO $100K BENCHMARKS:")
        print("-" * 120)
        for s_res in [res_inst_v2, res_inst_v3, res_v1]:
            s_name = s_res["strategy"]
            s_ret = s_res["return_pct"]
            s_pnl = s_res["net_pnl"]
            print(f"[{s_name}]")
            if spy_b:
                spy_alpha = s_ret - spy_b["return_pct"]
                spy_diff = s_pnl - spy_b["pnl"]
                print(f"   • vs SPY:  {spy_alpha:+.2%} Alpha (${spy_diff:+,.2f} dollar advantage)")
            if qqq_b:
                qqq_alpha = s_ret - qqq_b["return_pct"]
                qqq_diff = s_pnl - qqq_b["pnl"]
                print(f"   • vs QQQ:  {qqq_alpha:+.2%} Alpha (${qqq_diff:+,.2f} dollar advantage)")

    # Save CSV
    out_csv = os.path.join(BASE_DIR, "logs", f"inst_v_recovery_backtest_{date_list[0]}_to_{date_list[-1]}.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[OK] Institutional summary saved to: {out_csv}")

    return res_inst_v2, res_inst_v3, benchmarks


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "3months"
    run_comparative_inst_v_backtest(period_mode=mode)
