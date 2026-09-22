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
params = {"Ticker": Asset(symbol="MARA", asset_type=Asset.AssetType.STOCK), "Plot": False}

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

output_file = r"E:\repos\gotti-visualize\strategies\book_based_2\backtest_stats_obj_mara.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"| Strategy | CAGR | Total Return | Max Drawdown | Win Rate | Trades | Sharpe |\n")
    f.write(f"|----------|------|--------------|--------------|----------|--------|--------|\n")
    
    for name, cls in strategies:
        try:
            res = cls.backtest(
                YahooDataBacktesting,
                start, end,
                budget=10_000,
                parameters=params,
                show_plot=False
            )
            stats = res[0] if isinstance(res, tuple) else res
            
            cagr = stats.get('cagr', 'N/A')
            if isinstance(cagr, float): cagr = f"{cagr:.2%}"
            
            ret = stats.get('total_return', 'N/A')
            if isinstance(ret, float): ret = f"{ret:.2%}"
            
            dd = stats.get('max_drawdown', 'N/A')
            if isinstance(dd, float): dd = f"{dd:.2%}"
            
            win = stats.get('win_rate', 'N/A')
            if isinstance(win, float): win = f"{win:.2%}"
            
            trades = str(stats.get('total_trades', 'N/A'))
            
            sharpe = stats.get('sharpe_ratio', 'N/A')
            if isinstance(sharpe, float): sharpe = f"{sharpe:.2f}"
            
            f.write(f"| {name} | {cagr} | {ret} | {dd} | {win} | {trades} | {sharpe} |\n")
        except Exception as e:
            f.write(f"| {name} | Error | {str(e)[:30]} | - | - | - | - |\n")

print("Done.")
