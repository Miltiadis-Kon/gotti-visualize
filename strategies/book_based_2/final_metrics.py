import sys, os, io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

sys.path.append(r"E:\repos\gotti-visualize")

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
params = {"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK), "Plot": False}

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

output_file = r"E:\repos\gotti-visualize\strategies\book_based_2\backtest_final.txt"

class DevNull:
    def write(self, msg): pass
    def flush(self): pass

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"| Strategy | CAGR | Total Return | Max Drawdown | Win Rate | Trades | Sharpe |\n")
    f.write(f"|----------|------|--------------|--------------|----------|--------|--------|\n")
    
    for name, cls in strategies:
        try:
            with redirect_stdout(DevNull()), redirect_stderr(DevNull()):
                res = cls.backtest(
                    YahooDataBacktesting,
                    start, end,
                    budget=10_000,
                    parameters=params,
                    show_plot=False
                )
            stats = res[0] if isinstance(res, tuple) else res
            
            cagr = f"{stats.get('cagr', 0):.2%}" if stats.get('cagr') is not None else "N/A"
            ret = f"{stats.get('total_return', 0):.2%}" if stats.get('total_return') is not None else "N/A"
            dd = f"{stats.get('max_drawdown', 0):.2%}" if stats.get('max_drawdown') is not None else "N/A"
            win = f"{stats.get('win_rate', 0):.2%}" if stats.get('win_rate') is not None else "N/A"
            trades = str(stats.get('total_trades', 0))
            sharpe = f"{stats.get('sharpe_ratio', 0):.2f}" if stats.get('sharpe_ratio') is not None else "N/A"
            
            f.write(f"| {name} | {cagr} | {ret} | {dd} | {win} | {trades} | {sharpe} |\n")
        except Exception as e:
            f.write(f"| {name} | Error | {str(e)[:20]} | - | - | - | - |\n")

print("Done. Metrics saved to", output_file)
