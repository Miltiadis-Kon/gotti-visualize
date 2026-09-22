import sys
import io
import logging
from datetime import datetime
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.book_based_2.hcr_breakout_long import HCRBreakoutLong
from strategies.book_based_2.hcr_breakout_short import HCRBreakoutShort
from strategies.book_based_2.fib_retrace_swing import FibRetraceSwing
from strategies.book_based_2.gap_setup_long import OpeningGapLong
from strategies.book_based_2.gap_setup_short import OpeningGapShort
from strategies.book_based_2.retrace_long import RetraceLong
from strategies.book_based_2.retrace_short import RetraceShort
from strategies.book_based_2.rsi_setup import RSISetup

start = datetime(2025, 9, 12)
end = datetime(2026, 9, 12)

strategies = [
    ("HCR Breakout Long", HCRBreakoutLong),
    ("HCR Breakout Short", HCRBreakoutShort),
    ("Fib Retrace Swing", FibRetraceSwing),
    ("Gap Setup Long", OpeningGapLong),
    ("Gap Setup Short", OpeningGapShort),
    ("Retrace Long", RetraceLong),
    ("Retrace Short", RetraceShort),
    ("RSI Setup", RSISetup)
]

for ticker in ["NVDA", "PLTR", "MARA"]:
    print(f"\n--- {ticker} ---")
    for name, cls in strategies:
        try:
            log_capture_string = io.StringIO()
            ch = logging.StreamHandler(log_capture_string)
            ch.setLevel(logging.INFO)
            # Add to lumibot logger specifically
            logging.getLogger("lumibot").addHandler(ch)
            logging.getLogger().addHandler(ch)
            
            cls.backtest(
                YahooDataBacktesting,
                start, end,
                budget=10_000,
                parameters={"Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK), "Plot": False},
                show_plot=False
            )
            
            log_contents = log_capture_string.getvalue()
            buys = log_contents.count("Bought ")
            sells = log_contents.count("Sold ")
            
            logging.getLogger("lumibot").removeHandler(ch)
            logging.getLogger().removeHandler(ch)
            
            trades = max(buys, sells)
            
            print(f"{name:20s}: {trades} estimated trades (Buys: {buys}, Sells: {sells})")
            
        except Exception as e:
            print(f"{name:20s}: ERROR ({e})")
