import glob
import os
import re

html_files = glob.glob(r"E:\repos\gotti-visualize\logs\*_tearsheet.html")

for f in html_files:
    # Just look at the most recent one for RSI on MARA as a test
    if "RSISetup" in f:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            # In QuantStats HTML, look for "Trades" or similar
            # Or use regex to strip HTML tags and find the table
            text = re.sub('<[^<]+>', ' ', content)
            
            # Find the line containing "Total Trades" or just "Trades"
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if "Trades" in line or "trades" in line.lower():
                    print(f"Found in {os.path.basename(f)}: {line.strip()}")
            break
