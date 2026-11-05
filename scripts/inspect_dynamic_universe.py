import sys
import os

sys.path.insert(0, r"E:\repos\gotti-visualize")
from strategies.market_reactor import get_mysql_connection

conn = get_mysql_connection()
cur = conn.cursor(dictionary=True)
cur.execute("""
    SELECT signal_date, COUNT(DISTINCT ticker) as cnt 
    FROM signals 
    WHERE signal_date >= '2026-10-01' 
    GROUP BY signal_date 
    ORDER BY signal_date ASC
""")
rows = cur.fetchall()
print("Tickers with signals per day in October-November:")
for r in rows:
    print(f"  {r['signal_date']}: {r['cnt']} distinct tickers")

# Overall distinct tickers in that period
cur.execute("SELECT COUNT(DISTINCT ticker) as total FROM signals WHERE signal_date >= '2026-10-01'")
print(f"Total distinct tickers in period: {cur.fetchone()['total']}")

conn.close()
