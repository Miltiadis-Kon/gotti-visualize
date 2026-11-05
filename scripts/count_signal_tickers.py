import sys
import os
import pandas as pd

sys.path.insert(0, r"E:\repos\gotti-visualize")
from strategies.market_reactor import fetch_reactor_signals, get_mysql_connection

conn = get_mysql_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT DISTINCT signal_date FROM signals WHERE signal_date >= '2026-08-05' ORDER BY signal_date ASC")
dates = [r["signal_date"].strftime("%Y-%m-%d") for r in cursor.fetchall()]
conn.close()

active_tickers = set(["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR", "SMCI", "RBLX", "RIOT", "CLSK"])
high_cat_tickers = set(active_tickers)

for d in dates:
    sigs = fetch_reactor_signals(d, include_prev_afterhours=True)
    for s in sigs:
        r = float(getattr(s, "gemini_rating", 3.0) or 3.0)
        c = float(getattr(s, "news_conviction", 0.0) or 0.0)
        if (r >= 4.0 or r <= 2.0 or c >= 70.0) and s.ticker:
            high_cat_tickers.add(s.ticker.upper().strip())

print(f"High-catalyst tickers count: {len(high_cat_tickers)}")
print(f"Sample high-cat tickers: {sorted(list(high_cat_tickers))[:30]}")
