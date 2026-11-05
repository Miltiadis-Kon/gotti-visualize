import sys
import os
import pandas as pd
import numpy as np
from datetime import timedelta, date

sys.path.insert(0, r"E:\repos\gotti-visualize")
from strategies.market_reactor import get_mysql_connection
import yfinance as yf

def check_7day_consecutive_volatility(
    ticker: str,
    target_date: date,
    df_1d: pd.DataFrame,
    min_daily_range_pct: float = 0.020,
    lookback_days: int = 7,
):
    if df_1d is None or df_1d.empty:
        return False, 0.0, []
    
    prior_days = df_1d[df_1d.index.date < target_date]
    if len(prior_days) < lookback_days:
        return False, 0.0, []
    
    recent_7 = prior_days.iloc[-lookback_days:]
    daily_ranges = []
    
    for idx in range(len(recent_7)):
        row = recent_7.iloc[idx]
        h = float(row["high"])
        l = float(row["low"])
        rng = (h - l) / l if l > 0 else 0.0
        daily_ranges.append(rng)
        if rng < min_daily_range_pct:
            return False, float(np.mean(daily_ranges)), daily_ranges
            
    avg_rng = float(np.mean(daily_ranges))
    return True, avg_rng, daily_ranges

test_tickers = ["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR", "AAPL", "MSFT", "GOOGL", "RBLX", "ORCL", "SMCI"]
t_date = date(2026, 10, 28)

print(f"--- 7-DAY CONSECUTIVE VOLATILITY TEST FOR {t_date} ---")
for sym in test_tickers:
    df = yf.download(sym, start="2026-09-01", end="2026-10-29", interval="1d", progress=False)
    if not df.empty:
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        ok, avg_r, r_list = check_7day_consecutive_volatility(sym, t_date, df, min_daily_range_pct=0.020, lookback_days=7)
        status = "PASSED" if ok else "FAILED"
        print(f"[{status:6s}] {sym:5s} | 7D Avg Range: {avg_r:.2%} | Ranges: {[f'{x:.1%}' for x in r_list]}")
