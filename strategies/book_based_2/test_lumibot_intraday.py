from datetime import datetime, timedelta
import sys
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset
from strategies.book_based_2.retrace_long import RetraceLong

end = datetime(2026, 9, 12)
start = end - timedelta(days=59)

print(f"Running RetraceLong from {start.date()} to {end.date()}")

try:
    RetraceLong.backtest(
        YahooDataBacktesting,
        start, end,
        budget=10_000,
        parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK), "Plot": False},
        show_plot=False
    )
except Exception as e:
    print(f"Backtest error: {e}")
