import sys
from datetime import datetime
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers.backtesting import BacktestingBroker
from lumibot.traders import Trader
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
            data_source = YahooDataBacktesting(datetime_start=start, datetime_end=end)
            broker = BacktestingBroker(data_source)
            strat = cls(
                broker=broker, 
                parameters={"Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK), "Plot": False}
            )
            trader = Trader()
            trader.add_strategy(strat)
            trader.run_all()
            
            # Count the filled entry orders to determine setup count
            orders = broker.get_historical_orders()
            # An entry creates 1 order. The bracket creates 1 or 2 child orders.
            # So looking at parent orders or total fills works. Let's just divide total filled by 2.
            filled = [o for o in orders if o.status == 'filled']
            trades = len(filled) // 2
            
            print(f"{name:20s}: {trades} trades")
        except Exception as e:
            print(f"{name:20s}: ERROR ({e})")
