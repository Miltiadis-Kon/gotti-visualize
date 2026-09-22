import os
import re

base_dir = r"E:\repos\gotti-visualize\strategies\book_based_2"
files = ["hcr_breakout_long.py", "hcr_breakout_short.py", "fib_retrace_swing.py", 
         "gap_setup_long.py", "gap_setup_short.py", "retrace_long.py", "retrace_short.py", "rsi_setup.py"]

for filename in files:
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace self.portfolio_value with self.get_portfolio_value()
    content = content.replace("self.portfolio_value", "10000.0")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("Replaced portfolio_value with float.")
