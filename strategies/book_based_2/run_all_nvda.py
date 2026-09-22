import os, sys, io
from contextlib import redirect_stdout
from datetime import datetime

# Force UTF-8 encoding for stdout to prevent UnicodeEncodeError in Lumibot's progress bar
sys.stdout.reconfigure(encoding='utf-8')
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

params = {
    "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
    "Plot": False
}

strategies = [
    ("HCR_Breakout_Long", HCRBreakoutLong),
    ("HCR_Breakout_Short", HCRBreakoutShort),
    ("Fib_Retrace_Swing", FibRetraceSwing),
    ("Gap_Setup_Long", OpeningGapLong),
    ("Gap_Setup_Short", OpeningGapShort),
    ("Retrace_Long", RetraceLong),
    ("Retrace_Short", RetraceShort),
    ("RSI_Setup", RSISetup)
]

output_file = r"E:\repos\gotti-visualize\strategies\book_based_2\backtest_summary.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"NVDA Backtest Results (Last 1 Year: {start.date()} to {end.date()})\n")
    
    for name, cls in strategies:
        print(f"Running {name}...")
        f.write(f"\n{'='*50}\nSTRATEGY: {name}\n{'='*50}\n")
        
        # We need a custom StringIO that handles UTF-8 correctly
        class UTF8StringIO(io.StringIO):
            def write(self, s):
                if isinstance(s, bytes):
                    s = s.decode('utf-8')
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
                print(f"ERROR during backtest: {e}")
        
        full_out = out.getvalue()
        
        # Try to extract just the key tearsheet metrics
        lines = full_out.split('\n')
        recording = False
        summary_recorded = False
        for line in lines:
            if "Strategy   " in line or "CAGR" in line or "Total Return" in line:
                recording = True
            if recording:
                f.write(line + "\n")
                summary_recorded = True
            if recording and "Win Rate" in line:
                break
                
        if not summary_recorded:
            f.write(full_out)
            
        print(f"Finished {name}")

print(f"All backtests completed. Results saved to {output_file}")
