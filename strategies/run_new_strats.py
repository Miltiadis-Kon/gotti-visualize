import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Fix path to make sure strategies can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import AlpacaDataBacktesting
from lumibot.entities import Asset

# Load credentials
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
api_key = os.environ.get("APCA_API_KEY_PAPER")
api_secret = os.environ.get("APCA_API_SECRET_KEY_PAPER")

from strategies.pb_investing_vwap import PBInvestingVWAPEMA
from strategies.lance_breitstein_scalps import LanceBreitsteinScalping

# Past 6 months
end = datetime.now()
start = end - timedelta(days=180)

# We use AlpacaDataBacktesting which supports 5-minute data going back 5+ years
datasource_params = {
    "api_key": api_key,
    "api_secret": api_secret,
}

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
    print(f"Running 6-Month Backtests for {ticker} (Alpaca Data)...")
    print(f"===========================================\n")
    for name, cls, extra_params in strategies_to_test:
        params = {
            "Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK),
            "Plot": False
        }
        params.update(extra_params)
        
        try:
            # We ignore background WebSocket spam that lumibot might spawn by default
            results, strategy = cls.run_backtest(
                datasource_class=AlpacaDataBacktesting,
                datasource_parameters=datasource_params,
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
