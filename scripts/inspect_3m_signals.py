import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategies.market_reactor import get_mysql_connection

conn = get_mysql_connection()
cursor = conn.cursor(dictionary=True)

query = """
SELECT COUNT(1) as total_signals,
       MIN(signal_date) as min_date,
       MAX(signal_date) as max_date,
       COUNT(DISTINCT signal_date) as total_sessions
FROM signals
WHERE signal_date BETWEEN '2026-08-05' AND '2026-11-05'
"""
cursor.execute(query)
print("3-Month Signals Summary:", cursor.fetchone())

cursor.execute("""
SELECT DISTINCT signal_date
FROM signals
WHERE signal_date BETWEEN '2026-08-05' AND '2026-11-05'
ORDER BY signal_date ASC
""")
dates = [r["signal_date"].strftime("%Y-%m-%d") for r in cursor.fetchall()]
print(f"Total Trading Sessions: {len(dates)}")
print("First 5 sessions:", dates[:5])
print("Last 5 sessions:", dates[-5:])

cursor.execute("""
SELECT ticker, COUNT(1) as c
FROM signals
WHERE signal_date BETWEEN '2026-08-05' AND '2026-11-05'
GROUP BY ticker
ORDER BY c DESC
LIMIT 15
""")
print("Top 15 Tickers:")
for r in cursor.fetchall():
    print(f"  {r['ticker']}: {r['c']} signals")

conn.close()
