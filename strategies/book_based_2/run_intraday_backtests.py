import sys
from datetime import datetime, timedelta
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

end = datetime(2026, 9, 12)
start = end - timedelta(days=59)

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

print(f"Running 59-day (Yahoo max) 5-min intraday backtests for NVDA...")

for name, cls in strategies:
    try:
        results, strategy = cls.run_backtest(
            datasource_class=YahooDataBacktesting,
            backtesting_start=start,
            backtesting_end=end,
            budget=10_000,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK), "Plot": False},
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
            
        print(f"{name:20s}: Return: {ret:>6.2f}% | MaxDD: {mdd:>6.2f}% | Est. Trades: {trades}")
        
    except Exception as e:
        print(f"{name:20s}: ERROR ({e})")
