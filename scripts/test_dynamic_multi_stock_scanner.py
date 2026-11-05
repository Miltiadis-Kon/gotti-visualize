import sys
import os
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, r"E:\repos\gotti-visualize")
from strategies.market_reactor import fetch_reactor_signals, get_mysql_connection
from strategies.high_beta_catalyst_open import check_7day_consecutive_volatility
import yfinance as yf

# Test date with ~200 signals
test_date_str = "2026-10-29"
target_date = date(2026, 10, 29)

print(f"=== TESTING DYNAMIC MULTI-STOCK DISCOVERY FOR {test_date_str} ===")
sigs = fetch_reactor_signals(test_date_str, include_prev_afterhours=True)
all_tickers = sorted(list(set([s.ticker.upper().strip() for s in sigs if s.ticker])))
print(f"Total tickers with signals on {test_date_str}: {len(all_tickers)}")

# Filter top conviction/catalyst signals (conviction >= 75% or rating >= 3.5 or <= 2.5)
candidate_tickers = []
for sym in all_tickers:
    sym_sigs = [s for s in sigs if s.ticker == sym]
    latest = sym_sigs[-1]
    rating = float(getattr(latest, "gemini_rating", 3.0) or 3.0)
    conv = float(getattr(latest, "news_conviction", 0.0) or 0.0)
    if rating >= 3.5 or rating <= 2.5 or conv >= 75.0:
        candidate_tickers.append(sym)

print(f"Candidate tickers passing conviction filter: {len(candidate_tickers)}")

# Batch download 1d bars for these candidates from Alpaca or YFinance
start_1d = target_date - timedelta(days=20)
end_1d = target_date + timedelta(days=1)

qualified_stocks = []
tested_count = 0

for sym in candidate_tickers[:35]:  # test top 35
    try:
        df = yf.download(sym, start=start_1d.strftime("%Y-%m-%d"), end=end_1d.strftime("%Y-%m-%d"), interval="1d", progress=False)
        if not df.empty:
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            ok, avg_r, r_list = check_7day_consecutive_volatility(sym, target_date, df, min_daily_range_pct=0.020, lookback_days=7)
            tested_count += 1
            if ok:
                qualified_stocks.append((sym, avg_r, r_list))
                print(f"  [QUALIFIED] {sym:5s} | 7D Avg Range: {avg_r:.2%}")
    except Exception:
        pass

print(f"\nResult: Out of {tested_count} tested stocks, {len(qualified_stocks)} dynamically qualified!")
print("Qualified stocks:", [s[0] for s in qualified_stocks])
