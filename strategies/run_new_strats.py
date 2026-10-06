import sys
import os
from datetime import datetime, timedelta

# Prevent Alpaca from attempting WebSocket connections during Yahoo backtesting
os.environ["APCA_API_KEY_PAPER"] = ""
os.environ["APCA_API_SECRET_KEY_PAPER"] = ""

# Fix path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.pb_investing_vwap import PBInvestingVWAPEMA
from strategies.lance_breitstein_scalps import LanceBreitsteinScalping

# Past 5 days for Yahoo 1-minute data limitation
end = datetime.now()
start = end - timedelta(days=5)

strategies_to_test = [
    ("PB - VWAP Retest", PBInvestingVWAPEMA, {"Setup_Type": "vwap_retest"}),
    ("PB - PM Retest", PBInvestingVWAPEMA, {"Setup_Type": "pm_retest"}),
    ("PB - 8 EMA Scalp", PBInvestingVWAPEMA, {"Setup_Type": "ema_scalp"}),
    ("Lance - Breakout", LanceBreitsteinScalping, {"Strategy_Type": "breakout"}),
    ("Lance - Exhaustion", LanceBreitsteinScalping, {"Strategy_Type": "exhaustion"}),
]

tickers = ["PLTR", "NVDA", "AAPL"]

for ticker in tickers:
    print(f"\n===========================================")
    print(f"Running 5-Day Backtests for {ticker} (Yahoo Min Data)...")
    print(f"===========================================\n")
    for name, cls, extra_params in strategies_to_test:
        params = {
            "Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK),
            "Plot": False
        }
        params.update(extra_params)
        
        try:
            results, strategy = cls.run_backtest(
                datasource_class=YahooDataBacktesting,
                backtesting_start=start,
                backtesting_end=end,
                budget=20_000,
                parameters=params,
                show_plot=False
            )
            
            ret = results.get('total_return', 0) * 100
            
            mdd_raw = results.get('max_drawdown', 0)
            if isinstance(mdd_raw, dict):
                mdd = mdd_raw.get('drawdown', 0) * 100
            else:
                mdd = mdd_raw * 100
                
            try:
                orders = strategy.broker.get_historical_orders()
                filled = [o for o in orders if o.status == 'filled']
                trades = len(filled) // 2
            except:
                trades = "N/A"
                
            print(f"{name:22s}: Return: {ret:>7.2f}% | MaxDD: {mdd:>6.2f}% | Est. Trades: {trades}")
            
        except Exception as e:
            print(f"{name:22s}: ERROR ({e})")
