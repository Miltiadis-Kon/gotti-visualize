'''
System 1: Long Trend High Momentum
To be in trending stocks that have big momentum.
This gets you into the high-fliers, the popular stocks, when the market is an uptrend.
We only want to trade when the market sentiment is in our favor and we are in very liquid stocks.
I like a lot of volume for long-term positions, because the volume may diminish over time and you want to have a cushion so you can always get out with good liquidity.
In this system I want the trend of the stock to be up in a very simple way. 
 
From the book : Automated Stock Trading Systems by Lawrence Bensdorp
'''

import pandas_ta as ta
import pandas as pd


def format_date(date):
    '''
    Formats the date to the correct format
    :param date: type = string, description = Date string
    '''
    return date.strftime('%Y-%m-%d')


def filters(ticker_data, end_date):
    '''
    Average daily dollar volume greater than $50 million over the last twenty days.
    Minimum price $5.00.
    :param data: type = Dataframe, description = Ticker data
    :param end_date: type = string, description = Start date of filter 
    '''
    # Minimum price $5.00.
    if ticker_data['close'] < 5:
        return False
    
    end_date = pd.to_datetime(end_date)
    end_date_20 = end_date - pd.DateOffset(days=20)
    
    # Average daily dollar volume greater than $50 million over the last twenty days.
    df = ticker_data.loc[end_date_20:end_date] # Get the last 20 days as a DF
    avg_volume = df['volume'].mean() # Get mean value of volume
    if avg_volume < 50000000: # If less than 50 mil
        return False
    return True
    
    
#TODO: Migrate rank on screener
def rank():
    '''
    In case we have more setups than our position sizing allows,
    we rank by the highest rate of change over the last 200 trading days.
    This means the highest percentage price increase over the last 200 trading days. 
    '''
    pass

def setup(spy_data,ticker_data,end_date):
    '''
    Close of the SPY is above the 100-day simple moving average (SMA).
    This indicates a trend in the overall index.
    The close of the 25-day simple moving average is above the close of the 50-day simple moving average.
    '''

def entry():
    '''
    Next day market order on open.
    '''

def stop_loss():
    '''
    The day after entering the trade, place a stop-loss below the execution price of five times the average true range (ATR) of the last twenty days.
    '''

def reentry():
    '''
    If stopped out, reenter the next day if all entry conditions apply again.
    '''
    
def profit_protection():
    '''
    A trailing stop of 25 percent. This is in addition to the initial stop-loss.
    Eventually, as the stock rises, the trailing stop will move up above the stop-loss price.
    '''

def profit_taking():
    '''
    No profit target; the goal is to ride this as high as it will go.
    '''
    
def position_sizing():
    '''
    2 percent risk and 10 percent maximum percentage size, with a maximum of ten positions.
    '''
