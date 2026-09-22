import os

base_dir = r"E:\repos\gotti-visualize\strategies\book_based_2"

for filename in ["retrace_long.py", "retrace_short.py", "fib_retrace_swing.py", "rsi_setup.py", "hcr_breakout_long.py", "hcr_breakout_short.py", "gap_setup_long.py", "gap_setup_short.py"]:
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Add safety check after resample
    target = ".dropna()"
    replacement = ".dropna()\n        if len(df) < 15: return"
    content = content.replace(target, replacement)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Added length check.")
