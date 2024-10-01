# Import the yfinance library
import yfinance as yf
import mplfinance as mpf


# Function to get the ticker data
# timespan: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# time : 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
accepted_times = ['15m','1h','30m']

def get_ticker(stock_symbol, timespan, time):
    if not time in accepted_times:
        raise ValueError('Chosen interval is not available for analysis right now!')
    # Fetch data from yfinance
    ticker = yf.Ticker(stock_symbol)
    # Fetch historical data from yfinance
    data = ticker.history(period=timespan, interval=time)
    #Rename headers to lowercase
    data.columns = data.columns.str.lower()
    # Add datetime as a column
    data['date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data

def example():
    data = get_ticker('AAPL', '1d', '1m')
    print(data.head())
    mpf.plot(data, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')

#example() 
# Suppose that you have dataframe like the below.
#             date    open    high     low   close     volume
# 0     2018-12-31  244.92  245.54  242.87  245.28  147031456
# 1     2018-12-28  244.94  246.73  241.87  243.15  155998912
# 2     2018-12-27  238.06  243.68  234.52  243.46  189794032
# ...          ...     ...     ...     ...     ...        ...

