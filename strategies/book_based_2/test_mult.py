import sys
from datetime import datetime, timedelta
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.book_based_2.hcr_breakout_long import HCRBreakoutLong

end = datetime(2026, 9, 12)
start = end - timedelta(days=59)

results, strategy = HCRBreakoutLong.run_backtest(
    datasource_class=YahooDataBacktesting,
    backtesting_start=start,
    backtesting_end=end,
    budget=10_000,
    parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK), "Plot": False},
    show_plot=False
)

print(results)
print("Type of total_return:", type(results.get('total_return')))
try:
    print("Multiplication test:", results.get('total_return', 0) * 100)
except Exception as e:
    print("Multiplication ERROR:", e)
