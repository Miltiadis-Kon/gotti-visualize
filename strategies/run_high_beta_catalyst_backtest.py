"""
High-Beta Catalyst Opening Strategy Backtest Runner
===================================================
Simulates the 5-step rulebook, the 4 opening patterns (Golden V-Shape,
Straight Rally, Bear False Recovery, Forensic COIN Bull Flag Scale-In),
and the Catalyst Confluence Engine across high-beta tickers (COIN, NVDA, TSLA, AMD, PLTR, MARA, MSTR).

Compares results side-by-side with:
1. High-Beta Catalyst Open (Full 4-Pattern Engine + Scale-in)
2. News_LiquiditySweep
3. MarketReactor (Fade Mode)
4. SPY Benchmark ($100k Buy & Hold)
5. NASDAQ 100 / QQQ Benchmark ($100k Buy & Hold)
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
    HighBetaCatalystOpenStrategy,
)
from strategies.risk_management import calculate_atr
from strategies.run_combined_news_backtest import simulate_strategy_engine

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HighBetaBacktest")

# Target High-Beta Universe (moves 2%+ daily, high volatility)
HIGH_BETA_UNIVERSE = ["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR"]


def simulate_high_beta_catalyst_engine(
    date_list: List[str],
    all_eligible_signals: List[Tuple[str, ReactorSignal]],
    ticker_data_5m: Dict[str, pd.DataFrame],
    ticker_data_1d: Dict[str, pd.DataFrame],
    qqq_1d: Optional[pd.DataFrame],
    qqq_5m: Optional[pd.DataFrame],
    initial_portfolio: float = 100000.0,
    take_profit_pct: float = 0.020,     # Mechanical +2.0% Take Profit
    stop_loss_pct: float = 0.035,       # Mechanical -3.5% Stop Loss
    enable_scale_in: bool = True,       # 60/40 2-tranche scale-in on Pattern 4
    breakeven_gain_pct: float = 0.010,  # Ratchet stop to entry at +1.0% gain
) -> Dict[str, Any]:
    """
    Simulates the HighBetaCatalystOpen strategy execution across all dates in date_list.
    Enforces strict intraday execution: entries 09:30 - 10:15 EST, 15:55 EST flatten.
    """
    portfolio_cash = initial_portfolio
    closed_trades: List[Dict[str, Any]] = []
    pattern_stats: Dict[str, Dict[str, int]] = {
        "PATTERN_1_GOLDEN_V_SHAPE": {"trades": 0, "wins": 0, "losses": 0},
        "PATTERN_2_STRAIGHT_RALLY": {"trades": 0, "wins": 0, "losses": 0},
        "PATTERN_3_BEAR_FALSE_RECOVERY": {"trades": 0, "wins": 0, "losses": 0},
        "PATTERN_4_FORENSIC_COIN_BULL_FLAG": {"trades": 0, "wins": 0, "losses": 0},
    }

    for d_str in date_list:
        curr_date = pd.to_datetime(d_str).date()
        daily_sigs = [s for d, s in all_eligible_signals if d == d_str]

        # Market Regime on QQQ for curr_date (evaluated at market open)
        qqq_open = 480.0
        qqq_curr = 480.0
        prior_qqq_1d = qqq_1d[qqq_1d.index.date < curr_date] if qqq_1d is not None else None
        if qqq_5m is not None and not qqq_5m.empty:
            qqq_day = qqq_5m[qqq_5m.index.date == curr_date]
            if not qqq_day.empty:
                qqq_open = float(qqq_day.iloc[0]["open"])
                qqq_curr = float(qqq_day.iloc[1]["close"]) if len(qqq_day) >= 2 else qqq_open
        market_regime = CatalystConfluenceEngine.evaluate_market_regime(prior_qqq_1d, qqq_curr, qqq_open)

        # Cache daily signals for competitor catalyst analysis
        day_sig_cache = [s for _, s in all_eligible_signals if _ == d_str]

        # Iterate high-beta stocks that have signals or data
        seen_tickers: set = set()
        for ticker in HIGH_BETA_UNIVERSE:
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            if ticker not in ticker_data_5m:
                continue

            df_5m = ticker_data_5m[ticker]
            day_candles = df_5m[df_5m.index.date == curr_date]
            if len(day_candles) < 4:
                continue

            # Prior daily bars for ATR and PDC
            df_1d = ticker_data_1d.get(ticker)
            if df_1d is None or df_1d.empty:
                continue
            prior_1d = df_1d[df_1d.index.date < curr_date]
            if len(prior_1d) < 2:
                continue

            pdc = float(prior_1d.iloc[-1]["close"])
            daily_atr = float(calculate_atr(prior_1d, length=min(20, len(prior_1d) - 1))) or (pdc * 0.03)
            atr_pct = daily_atr / pdc
            if atr_pct < 0.020:  # Must move 2%+ daily
                continue

            # Step 1: Pre-Market Catalyst Confluence
            cat_analysis = CatalystConfluenceEngine.analyze_catalysts(
                target_ticker=ticker,
                target_date_str=d_str,
                signals_cache=day_sig_cache
            )
            cat_bias = cat_analysis["catalyst_bias"]
            cat_type = cat_analysis["catalyst_type"]

            # Opening parameters
            bar0 = day_candles.iloc[0]
            open_price = float(bar0["open"])
            session_low = float(day_candles.iloc[0]["low"])
            session_high = float(day_candles.iloc[0]["high"])

            trade_active = False
            trade_side = None
            entry_price = 0.0
            shares = 0
            stop_loss = 0.0
            take_profit = 0.0
            active_pattern = None
            tranche2_filled = False
            tranche1_shares = 0
            tranche2_shares = 0
            be_applied = False

            # Candle-by-candle simulation (5m bars)
            for idx in range(1, len(day_candles)):
                bar = day_candles.iloc[idx]
                bar_time = bar.name.time()
                cur_price = float(bar["close"])
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                session_low = min(session_low, bar_low)
                session_high = max(session_high, bar_high)

                prev_bar = day_candles.iloc[idx - 1]
                is_green_reversal = (bar["close"] > bar["open"]) and (bar["close"] > prev_bar["high"])
                is_red_breakdown = (bar["close"] < bar["open"]) and (bar["close"] < prev_bar["low"])

                # ── EVALUATE ENTRY (09:30 - 10:15 EST) ───────────────────────
                if not trade_active and time(9, 30) <= bar_time <= time(10, 15):
                    drop_from_open = (open_price - cur_price) / open_price
                    gain_from_open = (cur_price - open_price) / open_price

                    trigger_pattern = None
                    setup_side = "BUY"
                    setup_sl = 0.0
                    setup_tp = 0.0
                    reason = ""

                    # PATTERN 2: STRAIGHT RALLY (Bull Market + Bullish Catalyst > 4)
                    if market_regime == "BULL_MARKET" and cat_bias == "BULLISH" and bar_time <= time(9, 45):
                        if gain_from_open >= 0.008 and is_green_reversal:
                            trigger_pattern = "PATTERN_2_STRAIGHT_RALLY"
                            setup_side = "BUY"
                            setup_sl = min(cur_price * (1.0 - stop_loss_pct), session_low)
                            setup_tp = cur_price * (1.0 + take_profit_pct)
                            reason = f"Bull Market Inflow + {cat_type} (>4)"

                    # PATTERN 1: GOLDEN V-SHAPE (Bull Market + Neutral Catalyst -> PDC Rebound)
                    elif market_regime == "BULL_MARKET" and cat_bias == "NEUTRAL":
                        tested_pdc = (drop_from_open >= 0.008 and abs(cur_price - pdc) / pdc <= 0.008) or (cur_price <= pdc and drop_from_open >= 0.010)
                        if tested_pdc and is_green_reversal:
                            trigger_pattern = "PATTERN_1_GOLDEN_V_SHAPE"
                            setup_side = "BUY"
                            setup_sl = min(cur_price * (1.0 - stop_loss_pct), session_low - (0.002 * cur_price))
                            setup_tp = cur_price * (1.0 + take_profit_pct)
                            reason = "Bull Market Inflow + Non-Catalyst PDC Support Rebound"

                    # PATTERN 4: FORENSIC COIN V-RECOVERY + HIGHER-LOW BULL FLAG
                    # Strictly requires Bull Market + Bullish Catalyst Confluence (> 4 on stock or peers)
                    elif market_regime == "BULL_MARKET" and cat_bias == "BULLISH":
                        dip_depth = (open_price - session_low) / open_price
                        bounce_height = (session_high - session_low) / session_low
                        if dip_depth >= 0.008 and bounce_height >= 0.008:
                            is_higher_low = (bar_low >= session_low * 1.006) and (cur_price < session_high)
                            if is_higher_low and is_green_reversal:
                                trigger_pattern = "PATTERN_4_FORENSIC_COIN_BULL_FLAG"
                                setup_side = "BUY"
                                setup_sl = session_low - (0.003 * cur_price)
                                setup_tp = cur_price * (1.0 + take_profit_pct)
                                reason = f"Forensic V-Recovery Rebound off Higher Low (+{bounce_height:.1%}) | {cat_type} (>4)"

                    # PATTERN 3: BEAR MARKET FALSE RECOVERY BREAKDOWN
                    # (Bear Market Regime + Bearish Catalyst Confluence < 2 on stock or peers)
                    elif (market_regime == "BEAR_MARKET" or cat_bias == "BEARISH") and (cur_price < open_price):
                        attempted_recovery = (session_high < open_price * 1.003) and (session_high > session_low * 1.004)
                        breaks_morning_low = (cur_price < session_low) and is_red_breakdown
                        if attempted_recovery and breaks_morning_low:
                            trigger_pattern = "PATTERN_3_BEAR_FALSE_RECOVERY"
                            setup_side = "SELL"
                            setup_sl = min(cur_price * (1.0 + stop_loss_pct), session_high + (0.003 * cur_price))
                            setup_tp = cur_price * (1.0 - take_profit_pct)
                            reason = f"Bear Market Outflow + Failed Morning Recovery Breakdown | {cat_type} (<2)"

                    if trigger_pattern:
                        # Position sizing based on 2% risk budget
                        risk_per_share = abs(cur_price - setup_sl)
                        if risk_per_share > 0:
                            total_qty = floor((portfolio_cash * 0.02) / risk_per_share)
                            total_qty = max(2, min(total_qty, floor((portfolio_cash * 0.90) / cur_price)))

                            if enable_scale_in and trigger_pattern == "PATTERN_4_FORENSIC_COIN_BULL_FLAG":
                                tranche1_shares = floor(total_qty * 0.60)
                                tranche2_shares = total_qty - tranche1_shares
                                exec_qty = tranche1_shares
                            else:
                                exec_qty = total_qty
                                tranche1_shares = total_qty
                                tranche2_shares = 0

                            trade_active = True
                            trade_side = setup_side
                            entry_price = cur_price
                            shares = exec_qty
                            stop_loss = setup_sl
                            take_profit = setup_tp
                            active_pattern = trigger_pattern
                            be_applied = False
                            tranche2_filled = False
                            pattern_stats[trigger_pattern]["trades"] += 1
                            continue

                # ── ACTIVE TRADE MANAGEMENT ──────────────────────────────────
                if trade_active:
                    # Tranche 2 Scale-in for Pattern 4 (Bull Flag consolidation dip)
                    if (
                        enable_scale_in
                        and not tranche2_filled
                        and tranche2_shares > 0
                        and trade_side == "BUY"
                        and bar_time <= time(10, 30)
                    ):
                        if cur_price <= entry_price * 1.002 and cur_price > stop_loss:
                            entry_price = ((entry_price * tranche1_shares) + (cur_price * tranche2_shares)) / (tranche1_shares + tranche2_shares)
                            shares += tranche2_shares
                            tranche2_filled = True

                    # Breakeven Ratchet at +1.0%
                    gain_pct = (cur_price - entry_price) / entry_price if trade_side == "BUY" else (entry_price - cur_price) / entry_price
                    if not be_applied and gain_pct >= breakeven_gain_pct:
                        stop_loss = entry_price
                        be_applied = True

                    # Check TP and SL
                    exit_trade = False
                    exit_price = cur_price
                    exit_reason = ""

                    if trade_side == "BUY":
                        if bar_high >= take_profit:
                            exit_trade = True
                            exit_price = take_profit
                            exit_reason = "TAKE_PROFIT (+2.0%)"
                        elif bar_low <= stop_loss:
                            exit_trade = True
                            exit_price = stop_loss
                            exit_reason = "BREAKEVEN" if be_applied and exit_price >= entry_price else "STOP_LOSS (-3.5%)"
                    else:  # SELL
                        if bar_low <= take_profit:
                            exit_trade = True
                            exit_price = take_profit
                            exit_reason = "TAKE_PROFIT (+2.0%)"
                        elif bar_high >= stop_loss:
                            exit_trade = True
                            exit_price = stop_loss
                            exit_reason = "BREAKEVEN" if be_applied and exit_price <= entry_price else "STOP_LOSS (-3.5%)"

                    # Strict 15:55 EST Flatten
                    if not exit_trade and bar_time >= time(15, 55):
                        exit_trade = True
                        exit_price = cur_price
                        exit_reason = "EOD_FLATTEN"

                    if exit_trade:
                        pnl = (exit_price - entry_price) * shares if trade_side == "BUY" else (entry_price - exit_price) * shares
                        ret_pct = pnl / (entry_price * shares)
                        portfolio_cash += pnl
                        is_win = pnl > 0

                        if is_win:
                            pattern_stats[active_pattern]["wins"] += 1
                        else:
                            pattern_stats[active_pattern]["losses"] += 1

                        closed_trades.append({
                            "date": d_str,
                            "ticker": ticker,
                            "pattern": active_pattern,
                            "side": trade_side,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "shares": shares,
                            "pnl": pnl,
                            "return_pct": ret_pct,
                            "is_win": is_win,
                            "exit_reason": exit_reason,
                            "portfolio_value": portfolio_cash,
                        })
                        trade_active = False
                        break  # Max 1 trade per stock per day

    # Calculate overall metrics
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t["is_win"]]
    breakeven_trades = [t for t in closed_trades if t["exit_reason"] == "BREAKEVEN"]
    losing_trades = [t for t in closed_trades if not t["is_win"] and t["exit_reason"] != "BREAKEVEN"]
    win_rate = len(winning_trades) / (total_trades - len(breakeven_trades)) if (total_trades - len(breakeven_trades)) > 0 else 0.0

    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    net_pnl = portfolio_cash - initial_portfolio
    total_return_pct = net_pnl / initial_portfolio

    return {
        "strategy": "HighBeta_CatalystOpen (5-Step 4-Pattern Engine)",
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "breakeven_trades": len(breakeven_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net_pnl,
        "return_pct": total_return_pct,
        "final_portfolio": portfolio_cash,
        "pattern_stats": pattern_stats,
        "trades": closed_trades,
    }


def run_comparative_high_beta_backtest(period_mode: str = "2months"):
    """
    Runs high-beta backtest comparing HighBetaCatalystOpen against
    News_LiquiditySweep, MarketReactor (Fade), SPY, and QQQ.
    """
    print("\n" + "=" * 105)
    print(" HIGH-BETA CATALYST OPEN STRATEGY COMPARATIVE BACKTEST RUNNER")
    print("=" * 105)

    # Fetch signals from MySQL
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    min_query_date = '2026-08-05' if period_mode in ("3months", "3month", "6months", "6month") else '2026-09-01'
    cursor.execute(
        f"SELECT DISTINCT signal_date FROM signals WHERE signal_date >= '{min_query_date}' ORDER BY signal_date ASC"
    )
    all_dates = [r["signal_date"].strftime("%Y-%m-%d") for r in cursor.fetchall()]
    conn.close()

    if not all_dates:
        # Fallback date list
        all_dates = [
            "2026-08-29", "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-17",
            "2026-09-18", "2026-09-22", "2026-09-23", "2026-09-29", "2026-09-30",
            "2026-10-01", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16",
            "2026-10-19", "2026-10-20", "2026-10-28", "2026-10-29", "2026-11-02",
            "2026-11-03", "2026-11-05"
        ]

    if period_mode in ("3months", "3month"):
        date_list = [d for d in all_dates if d >= "2026-08-05"]
        period_title = "Past 3 Months (Aug 5 - Nov 5, 2026)"
    elif period_mode == "1month":
        date_list = [d for d in all_dates if d >= "2026-10-01"]
        period_title = "Past 1 Month (Oct 1 - Nov 5, 2026)"
    elif period_mode == "week":
        date_list = [d for d in all_dates if d >= "2026-10-28"]
        period_title = "Recent Sessions (Oct 28 - Nov 5, 2026)"
    else:  # "2months"
        date_list = [d for d in all_dates if d >= "2026-09-01"]
        period_title = "Past 2 Months (Sep 1 - Nov 5, 2026)"

    print(f"Evaluation Period: {period_title} ({len(date_list)} trading sessions)")
    print(f"High-Beta Focus Universe: {', '.join(HIGH_BETA_UNIVERSE)}")

    # Ingest signals for date list
    all_eligible: List[Tuple[str, ReactorSignal]] = []
    for d in date_list:
        sigs = fetch_reactor_signals(d, include_prev_afterhours=True)
        for s in sigs:
            all_eligible.append((d, s))

    print(f"Total News Signals Ingested: {len(all_eligible)}")

    # Download 5m and 1d data for High-Beta Universe and Benchmarks
    all_download_syms = list(set(HIGH_BETA_UNIVERSE + ["SPY", "QQQ"]))
    start_dt = pd.to_datetime(date_list[0]) - timedelta(days=7)
    end_dt = pd.to_datetime(date_list[-1]) + timedelta(days=2)

    print(f"Downloading 5-minute candles from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")

    ticker_data_5m: Dict[str, pd.DataFrame] = {}
    ticker_data_1d: Dict[str, pd.DataFrame] = {}

    # 1. Fetch 5m intraday bars via Alpaca IEX (supports > 60 days history)
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
                print(f"Alpaca successfully fetched 5m bars for {len(ticker_data_5m)} symbols.")
    except Exception as e:
        print(f"Alpaca 5m fetch notice: {e}")

    # 2. Fallback to YFinance for any missing symbols or 1d bars
    for sym in all_download_syms:
        try:
            if sym not in ticker_data_5m or ticker_data_5m[sym].empty:
                earliest_safe = pd.Timestamp.now().tz_localize(None) - timedelta(days=58)
                yf_start = max(start_dt, earliest_safe)
                df_5m = yf.download(sym, start=yf_start.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="5m", progress=False)
                if not df_5m.empty:
                    if isinstance(df_5m.columns, pd.MultiIndex):
                        df_5m.columns = [c[0].lower() for c in df_5m.columns]
                    else:
                        df_5m.columns = [c.lower() for c in df_5m.columns]
                    if df_5m.index.tz is not None:
                        df_5m.index = df_5m.index.tz_convert("America/New_York")
                    else:
                        df_5m.index = df_5m.index.tz_localize("UTC").tz_convert("America/New_York")
                    ticker_data_5m[sym] = df_5m

            # 1d bars (for daily ATR and PDC)
            df_1d = yf.download(sym, start="2026-05-01", end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)
            if not df_1d.empty:
                if isinstance(df_1d.columns, pd.MultiIndex):
                    df_1d.columns = [c[0].lower() for c in df_1d.columns]
                else:
                    df_1d.columns = [c.lower() for c in df_1d.columns]
                ticker_data_1d[sym] = df_1d
        except Exception as e:
            print(f"Notice fetching {sym} via YF: {e}")

    qqq_1d = ticker_data_1d.get("QQQ")
    qqq_5m = ticker_data_5m.get("QQQ")

    initial_portfolio = 100000.0

    # 1. Run HighBetaCatalystOpen Simulation
    res_high_beta = simulate_high_beta_catalyst_engine(
        date_list=date_list,
        all_eligible_signals=all_eligible,
        ticker_data_5m=ticker_data_5m,
        ticker_data_1d=ticker_data_1d,
        qqq_1d=qqq_1d,
        qqq_5m=qqq_5m,
        initial_portfolio=initial_portfolio,
        take_profit_pct=0.020,
        stop_loss_pct=0.035,
        enable_scale_in=True,
        breakeven_gain_pct=0.010,
    )

    # Resample 5m bars to 15m for LiquiditySweep & MarketReactor
    ticker_data_15m: Dict[str, pd.DataFrame] = {}
    for sym, df_5m in ticker_data_5m.items():
        try:
            df_15m = df_5m.resample("15min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            ticker_data_15m[sym] = df_15m
        except Exception:
            pass

    # 2. News_LiquiditySweep Simulation
    res_swp = simulate_strategy_engine(
        "News_LiquiditySweep", date_list, all_eligible, ticker_data_15m,
        all_market_dates=date_list, initial_portfolio=initial_portfolio
    )

    # 3. MarketReactor (Fade Mode) Simulation
    res_fade = simulate_strategy_engine(
        "MarketReactor", date_list, all_eligible, ticker_data_15m,
        all_market_dates=date_list, initial_portfolio=initial_portfolio,
        selector2_mode="fade"
    )
    res_fade["strategy"] = "MarketReactor (Fade Mode)"

    # 4. Benchmark Calculations ($100k allocation)
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

    # Display Side-by-Side Performance Matrix
    print("\n" + "=" * 115)
    print(f" SIDE-BY-SIDE PERFORMANCE MATRIX — {period_title.upper()}")
    print("=" * 115)

    summary_rows = [
        {
            "Strategy / Benchmark": res_high_beta["strategy"],
            "Total Trades": res_high_beta["total_trades"],
            "Wins": res_high_beta["winning_trades"],
            "Losses": res_high_beta["losing_trades"],
            "Win Rate": f"{res_high_beta['win_rate']:.1%}",
            "Profit Factor": f"{res_high_beta['profit_factor']:.2f}",
            "Gross Profit": f"${res_high_beta['gross_profit']:+,.2f}",
            "Gross Loss": f"-${res_high_beta['gross_loss']:,.2f}",
            "Net P&L": f"${res_high_beta['net_pnl']:+,.2f}",
            "Return %": f"{res_high_beta['return_pct']:+.2%}",
        },
        {
            "Strategy / Benchmark": res_swp["strategy"],
            "Total Trades": res_swp["total_trades"],
            "Wins": res_swp.get("tp_hits", res_swp.get("winning_trades", 0)),
            "Losses": res_swp.get("sl_hits", res_swp.get("losing_trades", 0)),
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
            "Win Rate": "100.0%" if b["pnl"] > 0 else "0.0%",
            "Profit Factor": "N/A",
            "Gross Profit": f"${max(0.0, b['pnl']):+,.2f}",
            "Gross Loss": f"-${abs(min(0.0, b['pnl'])):,.2f}",
            "Net P&L": f"${b['pnl']:+,.2f}",
            "Return %": f"{b['return_pct']:+.2%}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # Pattern Breakdown
    print("\n" + "-" * 115)
    print(" PATTERN-BY-PATTERN BREAKDOWN (CATALYST CONFLUENCE + OPENING REGIMES):")
    print("-" * 115)
    for pat_name, stats in res_high_beta["pattern_stats"].items():
        p_trades = stats["trades"]
        p_wins = stats["wins"]
        p_losses = stats["losses"]
        p_wr = (p_wins / p_trades) if p_trades > 0 else 0.0
        print(f" • {pat_name:<38}: {p_trades:>2} trades | {p_wins:>2} wins ({p_wr:.1%}) | {p_losses:>2} losses")

    # Alpha vs Benchmarks
    if benchmarks:
        spy_b = next((b for b in benchmarks if b["symbol"] == "SPY"), None)
        qqq_b = next((b for b in benchmarks if b["symbol"] == "QQQ"), None)
        print("\n" + "-" * 115)
        print(" ALPHA OUTPERFORMANCE RELATIVE TO $100K BENCHMARKS:")
        print("-" * 115)
        strat_ret = res_high_beta["return_pct"]
        strat_pnl = res_high_beta["net_pnl"]
        if spy_b:
            spy_alpha = strat_ret - spy_b["return_pct"]
            spy_diff = strat_pnl - spy_b["pnl"]
            print(f" • vs SPY (S&P 500):   {spy_alpha:+.2%} Alpha (${spy_diff:+,.2f} dollar advantage)")
        if qqq_b:
            qqq_alpha = strat_ret - qqq_b["return_pct"]
            qqq_diff = strat_pnl - qqq_b["pnl"]
            print(f" • vs QQQ (NASDAQ):   {qqq_alpha:+.2%} Alpha (${qqq_diff:+,.2f} dollar advantage)")

    # Save to logs
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    out_csv = os.path.join(BASE_DIR, "logs", f"high_beta_catalyst_backtest_{date_list[0]}_to_{date_list[-1]}.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[OK] High-beta summary saved to: {out_csv}")

    return res_high_beta, benchmarks


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "2months"
    run_comparative_high_beta_backtest(period_mode=mode)
