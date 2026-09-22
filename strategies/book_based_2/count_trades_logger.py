import sys
import logging
from datetime import datetime
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.book_based_2.gap_setup_short import OpeningGapShort
from strategies.book_based_2.retrace_long import RetraceLong

# Add a custom file handler to capture all lumibot logs
log_file = r"E:\repos\gotti-visualize\strategies\book_based_2\backtest_trade_logs.txt"
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(file_handler)

start = datetime(2025, 9, 12)
end = datetime(2026, 9, 12)

print("Running RetraceLong on PLTR...")
RetraceLong.backtest(
    YahooDataBacktesting,
    start, end,
    budget=10_000,
    parameters={"Ticker": Asset(symbol="PLTR", asset_type=Asset.AssetType.STOCK), "Plot": False},
    show_plot=False
)
print("Finished. Check the log file for trade frequency.")
