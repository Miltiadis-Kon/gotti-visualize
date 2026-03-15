import pandas as pd
import json
import csv
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def process_signals():
    # Load signals with proper quoting handling
    try:
        df = pd.read_csv('signals.csv', quoting=csv.QUOTE_MINIMAL, escapechar='\\', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    print(f"Successfully loaded {len(df)} signals.")
    
    # Sort by ticker then by date (oldest to newest)
    if 'signal_date' in df.columns and 'ticker' in df.columns:
        df['signal_date'] = pd.to_datetime(df['signal_date'], errors='coerce')
        df = df.dropna(subset=['signal_date']) # drop rows where date parsing failed
        df = df.sort_values(by=['ticker', 'signal_date'])
        
        # Group by ticker
        grouped = df.groupby('ticker')
        
        for ticker, group in grouped:
            print(f"\n=========================================")
            print(f"TICKER: {ticker}")
            print(f"=========================================")
            
            for _, row in group.iterrows():
                date_str = row['signal_date'].strftime('%Y-%m-%d')
                print(f"\n  Date: {date_str} | Position: {row.get('signal_position', 'N/A')}")
                print(f"  Signal ID: {row.get('signal_id', 'N/A')}")
                
                print(f"  Calendar Keys: {row.get('calendar_event_keys', '[]')}")
                print(f"  News Keys: {row.get('news_keys', '[]')}")
                print(f"  Fundamental Key: {row.get('fundamental_analysis_key', 'N/A')}")
                
                # Try to pretty print sentiment if it exists
                sentiment_val = row.get('sentiment', None)
                if pd.notna(sentiment_val):
                    try:
                        sentiment_data = json.loads(sentiment_val)
                        print(f"  Sentiment:")
                        for k, v in sentiment_data.items():
                            print(f"    - {k}: {v}")
                    except:
                        print(f"  Sentiment: {sentiment_val}")
                
                print(f"  -----------------------------------------")

if __name__ == '__main__':
    process_signals()
