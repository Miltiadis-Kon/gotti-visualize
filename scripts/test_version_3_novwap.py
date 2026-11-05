import sys
import os
from datetime import time, timedelta
import pandas as pd
import numpy as np
from math import floor

sys.path.insert(0, r"E:\repos\gotti-visualize")

from strategies.market_reactor import fetch_reactor_signals, ReactorSignal, get_mysql_connection
from strategies.high_beta_catalyst_open import COMPETITOR_MAP, CatalystConfluenceEngine
from strategies.institutional_v_recovery import compute_vwap, compute_ema
from strategies.risk_management import calculate_atr
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

HIGH_BETA_UNIVERSE = ["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR"]

conn = get_mysql_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT DISTINCT signal_date FROM signals WHERE signal_date >= '2026-08-05' ORDER BY signal_date ASC")
all_dates = [r["signal_date"].strftime("%Y-%m-%d") for r in cursor.fetchall()]
conn.close()

all_eligible = []
for d in all_dates:
    sigs = fetch_reactor_signals(d, include_prev_afterhours=True)
    for s in sigs:
        all_eligible.append((d, s))

alpaca_key = os.getenv("ALPACA_API_KEY")
alpaca_sec = os.getenv("ALPACA_API_SECRET")
alp_client = StockHistoricalDataClient(alpaca_key, alpaca_sec)

start_dt = pd.to_datetime(all_dates[0]) - timedelta(days=7)
end_dt = pd.to_datetime(all_dates[-1]) + timedelta(days=2)
all_download_syms = list(set(HIGH_BETA_UNIVERSE + ["SPY", "QQQ"]))

tf5 = TimeFrame(5, TimeFrameUnit.Minute)
req = StockBarsRequest(
    symbol_or_symbols=all_download_syms,
    timeframe=tf5,
    start=start_dt,
    end=end_dt,
    feed=DataFeed.IEX
)
bars_alp = alp_client.get_stock_bars(req)
df_alp = bars_alp.df

ticker_data_5m = {}
for sym in all_download_syms:
    if sym in df_alp.index.levels[0]:
        sub = df_alp.xs(sym).copy()
        sub.columns = [c.lower() for c in sub.columns]
        if sub.index.tz is not None:
            sub.index = sub.index.tz_convert("America/New_York")
        else:
            sub.index = sub.index.tz_localize("UTC").tz_convert("America/New_York")
        ticker_data_5m[sym] = sub

import yfinance as yf
ticker_data_1d = {}
for sym in all_download_syms:
    df_1d = yf.download(sym, start="2026-05-01", end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)
    if not df_1d.empty:
        df_1d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df_1d.columns]
        ticker_data_1d[sym] = df_1d

qqq_5m = ticker_data_5m.get("QQQ")
qqq_1d = ticker_data_1d.get("QQQ")

def run_v_version(date_subset, name, require_vwap=True, require_rvol=True, max_trades_per_day=2):
    portfolio_cash = 100000.0
    closed_trades = []

    for d_str in date_subset:
        curr_date = pd.to_datetime(d_str).date()
        day_sig_cache = [s for _, s in all_eligible if _ == d_str]
        qqq_day = qqq_5m[qqq_5m.index.date == curr_date] if qqq_5m is not None else None
        prior_qqq_1d = qqq_1d[qqq_1d.index.date < curr_date] if qqq_1d is not None else None

        if qqq_day is not None and len(qqq_day) > 0 and prior_qqq_1d is not None and len(prior_qqq_1d) >= 10:
            qqq_open = float(qqq_day.iloc[0]["open"])
            qqq_now = float(qqq_day.iloc[0]["close"])
            market_regime = CatalystConfluenceEngine.evaluate_market_regime(
                prior_qqq_1d, qqq_now, qqq_open
            )
            if market_regime != "BULL_MARKET":
                continue

        day_candidates = []
        seen_tickers = set()

        for ticker in HIGH_BETA_UNIVERSE:
            if ticker in seen_tickers or ticker not in ticker_data_5m:
                continue
            seen_tickers.add(ticker)

            df_5m = ticker_data_5m[ticker]
            day_candles = df_5m[df_5m.index.date == curr_date].copy()
            if len(day_candles) < 4:
                continue

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

            cat_analysis = CatalystConfluenceEngine.analyze_catalysts(
                target_ticker=ticker,
                target_date_str=d_str,
                signals_cache=day_sig_cache
            )
            cat_bias = cat_analysis["catalyst_bias"]

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

                if time(9, 35) <= b_time <= time(10, 15):
                    vwap_ok = True if not require_vwap else ((cur_price >= cur_vwap) or (bar_low <= cur_vwap and cur_price > cur_vwap))
                    vol_ma20 = float(day_candles["volume"].iloc[:idx+1].rolling(min(20, idx+1)).mean().iloc[-1])
                    cur_vol = float(bar["volume"])
                    rvol = (cur_vol / vol_ma20) if vol_ma20 > 0 else 1.0
                    rvol_ok = True if not require_rvol else ((rvol >= 1.5) or (cur_vol >= vol_ma20 * 1.3))

                    has_bullish_div = True
                    if qqq_day is not None and len(qqq_day) > idx:
                        qqq_bar = qqq_day.iloc[idx]
                        qqq_prev = qqq_day.iloc[idx - 1]
                        qqq_dropping = float(qqq_bar["low"]) <= float(qqq_prev["low"])
                        stock_holding = bar_low >= prev_bar["low"]
                        has_bullish_div = (not qqq_dropping) or stock_holding

                    drop_from_open = (open_price - session_low) / open_price
                    bounce_height = (cur_price - session_low) / max(0.01, session_low)

                    if (
                        is_green_reversal
                        and vwap_ok
                        and rvol_ok
                        and has_bullish_div
                        and (cat_bias == "BULLISH" or drop_from_open >= 0.010)
                        and bounce_height >= 0.006
                    ):
                        atr_buffer = 1.2 * (daily_atr * 0.20)
                        hard_sl = max(session_low - atr_buffer, cur_price * 0.970)
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
                            })
                            break

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
            target1_price = entry_price * 1.020
            be_applied = False

            for idx in range(start_idx + 1, len(candles)):
                bar = candles.iloc[idx]
                b_time = bar.name.time()
                cur_price = float(bar["close"])
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                cur_ema9 = float(bar["ema9"])
                prev_bar = candles.iloc[idx - 1]

                gain_pct = (cur_price - entry_price) / entry_price

                if not be_applied and gain_pct >= 0.010:
                    stop_loss = entry_price + (0.001 * entry_price)
                    be_applied = True

                if not tier1_sold and bar_high >= target1_price:
                    half_qty = floor(total_shares * 0.50)
                    tier1_gain = (target1_price - entry_price) * half_qty
                    tier1_pnl = tier1_gain
                    remaining_shares -= half_qty
                    tier1_sold = True
                    stop_loss = max(stop_loss, entry_price)

                trade_closed = False
                exit_price = cur_price
                exit_reason = ""

                if bar_low <= stop_loss:
                    trade_closed = True
                    exit_price = stop_loss
                    exit_reason = "BREAKEVEN" if be_applied and exit_price >= entry_price else "STOP_LOSS"

                elif tier1_sold and bar_high >= (entry_price * 1.035):
                    trade_closed = True
                    exit_price = entry_price * 1.035
                    exit_reason = "RUNNER_TARGET"

                elif tier1_sold and cur_price < cur_ema9 and prev_bar["close"] < cur_ema9 and b_time >= time(10, 30):
                    trade_closed = True
                    exit_price = cur_price
                    exit_reason = "EMA9_TRAIL"

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
                        "date": d_str, "ticker": ticker, "pnl": total_pnl, "is_win": is_win,
                        "exit_reason": exit_reason, "ret": trade_ret, "entry": entry_price, "exit": exit_price
                    })
                    break

    tot = len(closed_trades)
    wins = [t for t in closed_trades if t["is_win"]]
    losses = [t for t in closed_trades if not t["is_win"] and abs(t["pnl"]) >= 5.0]
    bes = [t for t in closed_trades if abs(t["pnl"]) < 5.0 or "BREAKEVEN" in t["exit_reason"]]
    eff = tot - len(bes)
    wr = len(wins) / eff if eff > 0 else 0
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gp / gl if gl > 0 else 999.0
    net = portfolio_cash - 100000.0

    print(f"{name:45s} | Trades: {tot:2d} (W:{len(wins):2d} L:{len(losses):2d} BE:{len(bes):2d}) | WR: {wr:5.1%} | PF: {pf:4.2f} | GP: ${gp:+9.2f} | GL: -${gl:8.2f} | Net: ${net:+9.2f} ({net/1000:.2f}%)")
    return {
        "trades": tot, "wins": len(wins), "losses": len(losses), "breakeven": len(bes),
        "win_rate": wr, "pf": pf, "gp": gp, "gl": gl, "net": net
    }

print("\n" + "=" * 115)
print("--- 3-MONTH WINDOW (AUG 5 - NOV 5, 2026) ---")
run_v_version(all_dates, "V2: WITH VWAP & RVOL (Institutional)", require_vwap=True, require_rvol=True)
run_v_version(all_dates, "V3: WITHOUT VWAP & RVOL", require_vwap=False, require_rvol=False)

dates_2m = [d for d in all_dates if d >= "2026-09-01"]
print("\n--- 2-MONTH WINDOW (SEP 1 - NOV 5, 2026) ---")
run_v_version(dates_2m, "V2: WITH VWAP & RVOL (Institutional)", require_vwap=True, require_rvol=True)
run_v_version(dates_2m, "V3: WITHOUT VWAP & RVOL", require_vwap=False, require_rvol=False)

dates_1m = [d for d in all_dates if d >= "2026-10-01"]
print("\n--- 1-MONTH WINDOW (OCT 1 - NOV 5, 2026) ---")
run_v_version(dates_1m, "V2: WITH VWAP & RVOL (Institutional)", require_vwap=True, require_rvol=True)
run_v_version(dates_1m, "V3: WITHOUT VWAP & RVOL", require_vwap=False, require_rvol=False)
