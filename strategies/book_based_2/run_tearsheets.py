import sys, os, io
from contextlib import redirect_stdout
from datetime import datetime

sys.path.append(r"E:\repos\gotti-visualize")

# Monkeypatch the progress bar to completely silence it and prevent encoding errors
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

output_file = r"E:\repos\gotti-visualize\strategies\book_based_2\backtest_tearsheets.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for name, cls in strategies:
        print(f"Running {name}...")
        f.write(f"\n=================================\n{name}\n=================================\n")
        
        # We use a custom stdout that forces utf8 and stores to memory
        class UTF8StringIO(io.StringIO):
            def write(self, s):
                if isinstance(s, bytes): s = s.decode('utf-8', errors='ignore')
                super().write(s)
                
        out = UTF8StringIO()
        with redirect_stdout(out):
            try:
                cls.backtest(
                    YahooDataBacktesting,
                    start, end,
                    budget=10_000,
                    parameters=params,
                    show_plot=False
                )
            except Exception as e:
                print(f"Error: {e}")
                
        text = out.getvalue()
        
        # Filter text to just the tearsheet part
        recording = False
        summary_recorded = False
        for line in text.split('\n'):
            if "Strategy   " in line or "Total Return" in line or "CAGR" in line:
                recording = True
            if recording:
                f.write(line + "\n")
                summary_recorded = True
            if recording and "Win Rate" in line:
                break
                
        if not summary_recorded:
            f.write(text)

print("Finished!")
