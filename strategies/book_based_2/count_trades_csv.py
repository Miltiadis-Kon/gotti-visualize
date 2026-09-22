import sys
import os
import glob
import pandas as pd
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
            # We use save_logfile=True so Lumibot dumps trades.csv to disk
            res = cls.backtest(
                YahooDataBacktesting,
                start, end,
                budget=10_000,
                parameters={"Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK), "Plot": False},
                show_plot=False,
                save_logfile=True
            )
            
            # Now find the latest trades.csv in logs
            logs_dir = r"E:\repos\gotti-visualize\logs"
            # It creates a folder with the class name
            search_str = os.path.join(logs_dir, f"{cls.__name__}_*", "trades.csv")
            trade_files = glob.glob(search_str)
            if not trade_files:
                print(f"{name:20s}: 0 trades (no CSV)")
                continue
                
            # sort by modification time to get the one we just ran
            latest_file = max(trade_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            if df.empty or 'status' not in df.columns:
                print(f"{name:20s}: 0 trades")
            else:
                filled = df[df['status'] == 'filled']
                # Each round trip is an entry and an exit = 2 orders. 
                trades = len(filled) // 2
                print(f"{name:20s}: {trades} trades")
                
        except Exception as e:
            print(f"{name:20s}: ERROR ({e})")
