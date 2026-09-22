import os
import re

base_dir = r"E:\repos\gotti-visualize\strategies\book_based_2"

files = ["hcr_breakout_long.py", "hcr_breakout_short.py", "fib_retrace_swing.py", 
         "gap_setup_long.py", "gap_setup_short.py", "retrace_long.py", "retrace_short.py", "rsi_setup.py"]

for filename in files:
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace the bar fetching logic
    content = re.sub(r'get_historical_prices\(symbol, \d+, "minute"\)', 
                     r'get_historical_prices(symbol, 200, "minute")', content)
                     
    # Replace the first df length check to be safe
    content = re.sub(r'if bars is None or len\(bars\.df\) < \d+: return',
                     r'if bars is None or len(bars.df) < 50: return', content)
                     
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("Updated all strategies to fetch 200 minute bars.")
