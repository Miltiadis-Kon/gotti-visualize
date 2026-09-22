import yfinance as yf
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=90)

try:
    df = yf.download("NVDA", start=start, end=end, interval="5m")
    print(f"5m interval from 90 days ago: {len(df)} rows")
    if len(df) > 0:
        print(f"First index: {df.index[0]}")
except Exception as e:
    print(f"Error 5m: {e}")

try:
    df_1m = yf.download("NVDA", start=end - timedelta(days=7), end=end, interval="1m")
    print(f"\n1m interval from 7 days ago: {len(df_1m)} rows")
except Exception as e:
    print(f"Error 1m: {e}")
