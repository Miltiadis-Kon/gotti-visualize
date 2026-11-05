import sys
import os
import pandas as pd
from datetime import timedelta

sys.path.insert(0, r"E:\repos\gotti-visualize")
from strategies.market_reactor import get_mysql_connection

conn = get_mysql_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT DISTINCT ticker FROM signals WHERE signal_date >= '2026-08-05'")
db_tickers = [r["ticker"].upper().strip() for r in cursor.fetchall() if r["ticker"]]
cursor.execute("SELECT DISTINCT signal_date FROM signals WHERE signal_date >= '2026-08-05' ORDER BY signal_date ASC")
dates = [r["signal_date"].strftime("%Y-%m-%d") for r in cursor.fetchall()]
conn.close()

print(f"Total distinct tickers in signals table: {len(db_tickers)}")
print(f"Sample tickers: {db_tickers[:20]}")

import yfinance as yf
# Test 7-day consecutive volatility for a sample of tickers
sample = list(set(["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR"] + db_tickers[:25]))
start_dt = pd.to_datetime(dates[0]) - timedelta(days=20)
end_dt = pd.to_datetime(dates[-1]) + timedelta(days=2)

df_all = yf.download(sample, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)
if isinstance(df_all.columns, pd.MultiIndex):
    df_all.columns = [c[0].lower() + "_" + c[1] for c in df_all.columns]
else:
    df_all.columns = [c.lower() for c in df_all.columns]

print(f"Downloaded daily data for {len(sample)} tickers.")
