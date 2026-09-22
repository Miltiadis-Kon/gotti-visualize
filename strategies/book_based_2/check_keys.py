import sys
from datetime import datetime
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.book_based_2.gap_setup_short import OpeningGapShort

start = datetime(2025, 9, 12)
end = datetime(2026, 9, 12)
params = {"Ticker": Asset(symbol="MARA", asset_type=Asset.AssetType.STOCK), "Plot": False}

res = OpeningGapShort.backtest(
    YahooDataBacktesting,
    start, end,
    budget=10_000,
    parameters=params,
    show_plot=False
)
stats = res[0] if isinstance(res, tuple) else res
print("KEYS:", list(stats.keys()))
print("Total Trades from stats:", stats.get("total_trades"))

# We can also check the length of the positions or trades list if it's there
if 'trades' in stats: print("Number of trades (stats['trades']):", len(stats['trades']))

# Or just use the total number of orders
if hasattr(OpeningGapShort, 'get_orders'):
    print("This doesn't work post-backtest easily without the instance")
