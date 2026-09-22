import sys
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
            results, strategy = cls.run_backtest(
                datasource_class=YahooDataBacktesting,
                backtesting_start=start,
                backtesting_end=end,
                budget=10_000,
                parameters={"Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK), "Plot": False},
                show_plot=False
            )
            
            # The broker holds the trades!
            orders = strategy.broker.get_historical_orders()
            # Fills only
            filled = [o for o in orders if o.status == 'filled']
            
            # Group by order or just count entries
            # A completed setup usually has 1 entry order. Bracket exit orders will also fill.
            # Total fills / 2 gives a rough setup count, or we can just count the orders with side='buy' and side='sell'
            entries = sum(1 for o in filled if not o.is_bracket_child) # assuming bracket child is true for exits if configured that way
            # Actually easiest is just filled // 2
            trades = len(filled) // 2
            
            print(f"{name:20s}: {trades} round-trip trades (total filled orders: {len(filled)})")
        except Exception as e:
            print(f"{name:20s}: ERROR ({e})")
