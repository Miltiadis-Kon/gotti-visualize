# Import the yfinance library
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Function to get the ticker data
# timespan: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# time : 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
accepted_times = ['5m','15m','1h','30m']

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


def fetch_and_save_stock_data(ticker, start_date, end_date, filename):
    # Fetch the stock data
    stock = yf.Ticker(ticker)
    data = stock.history(period='5d', interval="5m")
    
    # Save the data to a CSV file
    data.to_csv(filename)
    print(f"Data saved to {filename}")


def sort_and_filter_market_cap(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert 'Market Cap' to numeric, errors='coerce' will turn non-numeric values into NaN
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

    # Drop rows with NaN values in 'Market Cap'
    df = df.dropna(subset=['Market Cap'])

    # Sort the DataFrame by 'Market Cap' in descending order
    df_sorted = df.sort_values(by='Market Cap', ascending=False)

    return df_sorted

def categorize_by_market_cap(df):
    # Define market cap categories
    categories = {
        'Large Cap': (10**10, float('inf')),
        'Mid Cap': (2*10**9, 10**10),
        'Small Cap': (300*10**6, 2*10**9),
        'Micro Cap': (50*10**6, 300*10**6),
        'Nano Cap': (0, 50*10**6)
    }

    # Create a dictionary to hold categorized DataFrames
    categorized_df = {}
    for category, (min_cap, max_cap) in categories.items():
        categorized_df[category] = df[(df['Market Cap'] >= min_cap) & (df['Market Cap'] < max_cap)]

    return categorized_df 

def sort_by_sector(df):
    sectors = df['Sector'].unique()
    sector_df = {}
    for sector in sectors:
        sector_df[sector] = df[df['Sector'] == sector]
    return sector_df

def sort_by_volume(sectors):
    # Sort each sector DataFrame by 'Volume' in descending order
    sorted_sectors = {}
    for sector, df in sectors.items():
        sorted_sectors[sector] = df.sort_values(by='Volume', ascending=False)
    return sorted_sectors

def sort_by_abs_percent_change(sectors):
    # Sort each sector DataFrame by absolute '% Change' in descending order
    sorted_sectors = {}
    for sector, df in sectors.items():
        df['% Change'] = df['% Change'].str.rstrip('%').astype('float')
        sorted_sectors[sector] = df.sort_values(by='% Change', key=abs, ascending=False)
    return sorted_sectors

def remove_microcaps(df):
    '''
    Remove stock with Market Cap < 1B, as penny stocks perform unpredictable!
    '''
    if 'Market Cap' in df.columns:
        df = df[df['Market Cap'] >= 10**9]
    return df

def remove_low_vol(df):
    '''
    To reduce such risk, 
    it's best to stick with stocks that have a minimum dollar volume of
    $20 million to $25 million. 
    But we are risky too so we will go a bit lower!
    $2 million
    '''
    if 'Volume' in df.columns:
        df = df[df['Volume'] >= 2*10**6]
    return df

def screen_all(file_path):
    df = pd.read_csv(file_path)
    # Convert 'Market Cap' to numeric, errors='coerce' will turn non-numeric values into NaN
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    # Drop rows with NaN values in 'Market Cap'
    df = df.dropna(subset=['Market Cap'])
    #Drop IPO year and country
    df = df.drop(columns=['IPO Year', 'Country'])
    #Further data processing
    df = remove_microcaps(df) # Comment this in order to uncomment Micro,Nano Caps
    df = remove_low_vol(df)
    #Sort by sector
    sectors = sort_by_sector(df)
    #Categorize each sector by market cap(Large Cap,Mid Cap etc) & sort by volume ( high volume = high payoff)
    for key in sectors.keys():
        sectors[key] = categorize_by_market_cap(sectors[key])
        sectors[key] = sort_by_volume(sectors[key])
    return sectors

def present_screener(screener):
    # Create lists to hold table data
    sectors_list = []
    categories_list = []
    tickers_list = []
    market_caps_list = []
    volumes_list = []
    percent_changes_list = []

    for sector, categories in screener.items():
        for category, df in categories.items():
            for _, row in df.iterrows():
                sectors_list.append(sector)
                categories_list.append(category)
                tickers_list.append(row['Symbol'])
                market_caps_list.append(row['Market Cap'])
                volumes_list.append(row['Volume'])
                percent_changes_list.append(row['% Change'])

    # Create a Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Sector", "Category", "Ticker", "Market Cap", "Volume", "% Change"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[sectors_list, categories_list, tickers_list, market_caps_list, volumes_list, percent_changes_list],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title='Stock Screener Results')
    fig.show()

def print_screener_length(screener):
    total_length = 0
    for sector, categories in screener.items():
        for category, df in categories.items():
            total_length += len(df)
    print(f"Total number of rows in screener: {total_length}")


def get_screened_tickers(screener):
    tickers = []
    for sector, categories in screener.items():
        for category, df in categories.items():
            tickers.extend(df['Symbol'].tolist())
    return tickers



# Example usage
def example():
    screener = screen_all('nasdaq_screener_all.csv')
    print_screener_length(screener)
    tickers = get_screened_tickers(screener)
    for ticker in tickers:
        data = get_ticker(ticker, '5d', '5m')
        #post_process(data) # Add your post-processing logic here (breakout pivot points etc..)
                        #add data to DB



'''
OUTPUT DATA FORMAT:
{'Sector': 
    {'Large Cap': Symbol  ... Industry},
    {'Small Cap' : Symbol ... Industry}
}
'''


'''
# Example usage
ticker_symbol = "AAPL"  # Apple Inc.
end_date = datetime.now()
start_date = end_date - timedelta(days=59)  # Last 365 days

filename = f"{ticker_symbol}_stock_data_5m_5d.csv"

fetch_and_save_stock_data(ticker_symbol, start_date, end_date, filename)
'''



# Suppose that you have dataframe like the below.
#             date    open    high     low   close     volume
# 0     2018-12-31  244.92  245.54  242.87  245.28  147031456
# 1     2018-12-28  244.94  246.73  241.87  243.15  155998912
# 2     2018-12-27  238.06  243.68  234.52  243.46  189794032
# ...          ...     ...     ...     ...     ...        ...

