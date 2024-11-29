import pandas as pd
from lightweight_charts import Chart


if __name__ == '__main__':
    
    chart = Chart()
    
    # Columns: time | open | high | low | close | volume 
    df = pd.read_csv('data\AAPL_stock_data_1d_1y.csv')
    
    # True columns: time | open | high | low | close | volume
    df.columns = df.columns.str.lower()
    df = df.drop(columns=['dividends', 'stock splits'])
    
    # Rename columns
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # Change this 2024-09-03 11:00:00-04:00 to Timestamp
    df['time'] = pd.to_datetime(df['time'],utc=True)
    
    
    chart.set(df)
    
    chart.show(block=True)
