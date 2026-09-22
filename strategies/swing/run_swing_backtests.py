import sys
from datetime import datetime, timedelta
sys.path.append(r"E:\repos\gotti-visualize")

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.swing.hcr_breakout_long import HCRBreakoutLongSwing
from strategies.swing.hcr_breakout_short import HCRBreakoutShortSwing
from strategies.swing.fib_retrace_swing import FibRetraceSwing
from strategies.swing.gap_setup_long import GapSetupLongSwing
from strategies.swing.gap_setup_short import GapSetupShortSwing
from strategies.swing.retrace_long import RetraceLongSwing
from strategies.swing.retrace_short import RetraceShortSwing
from strategies.swing.rsi_setup import RSISetupSwing

end = datetime(2026, 9, 12)
start = end - timedelta(days=365)

strategies = [
    ("HCR Breakout Long", HCRBreakoutLongSwing),
    ("HCR Breakout Short", HCRBreakoutShortSwing),
    ("Fib Retrace Swing", FibRetraceSwing),
    ("Gap Setup Long", GapSetupLongSwing),
    ("Gap Setup Short", GapSetupShortSwing),
    ("Retrace Long", RetraceLongSwing),
    ("Retrace Short", RetraceShortSwing),
    ("RSI Setup", RSISetupSwing)
]

for ticker in ["NVDA", "MARA", "PLTR"]:
    print(f"\n===========================================")
    print(f"Running 1-Year Daily Swing Backtests for {ticker}...")
    print(f"===========================================\n")
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
                
            print(f"{name:20s}: Return: {ret:>7.2f}% | MaxDD: {mdd:>6.2f}% | Est. Trades: {trades}")
            
        except Exception as e:
            print(f"{name:20s}: ERROR ({e})")
