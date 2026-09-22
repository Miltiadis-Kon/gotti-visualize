import requests
from bs4 import BeautifulSoup
import re

urls = [
    "https://thepatternsite.com/hcrl.html",
    "https://thepatternsite.com/hcrs.html",
    "https://thepatternsite.com/Daytrade.html",
    "https://thepatternsite.com/gapsetupl.html",
    "https://thepatternsite.com/gapsetups.html",
    "https://thepatternsite.com/retracel.html",
    "https://thepatternsite.com/retraces.html",
    "https://thepatternsite.com/rsisetup.html"
]

headers = {'User-Agent': 'Mozilla/5.0'}

for url in urls:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        
        # Look for keywords around 'chart', 'minute', 'min', 'daily'
        matches = []
        for line in text.split('\n'):
            line_lower = line.lower()
            if 'min' in line_lower or 'chart' in line_lower or 'daily' in line_lower or 'time' in line_lower:
                if '5' in line_lower or '15' in line_lower or '1' in line_lower or 'daily' in line_lower:
                    matches.append(line.strip()[:150]) # truncate long lines
        
        print(f"\nURL: {url}")
        if matches:
            # deduplicate
            unique_matches = list(set(matches))
            for m in unique_matches[:10]: # Print first few relevant lines
                print(f" - {m}")
        else:
            print(" - No obvious timeframe mentions found.")
            
    except Exception as e:
        print(f"Error fetching {url}: {e}")
