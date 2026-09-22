import sys
import traceback
from datetime import datetime, timedelta
sys.path.append(r"E:\repos\gotti-visualize")

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset
from strategies.book_based_2.hcr_breakout_long import HCRBreakoutLong

end = datetime(2026, 9, 12)
start = end - timedelta(days=59)

try:
    HCRBreakoutLong.run_backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=start,
        backtesting_end=end,
        budget=10_000,
        parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK), "Plot": False},
        show_plot=False
    )
except Exception as e:
    traceback.print_exc()
