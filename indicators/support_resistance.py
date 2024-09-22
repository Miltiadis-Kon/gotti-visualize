'''
Description: This script will help you to find the support and resistance levels of a stock using Python.
The script will fetch the historical data of a stock using the yfinance library and then calculate the support and resistance levels.
The support and resistance levels are calculated using the rolling minimum and maximum functions in pandas.
The script will also identify the major support and resistance levels based on the frequency of the support and resistance levels.
The support and resistance levels are then plotted on a candlestick chart using the mplfinance library.
To use call 

    df_daily,major_support, major_resistance = find_support_resistance_levels('AAPL', '5d', '15m')
    plot_support_resistance(df_daily, major_support, major_resistance)
    
MADE BY: KONTOS MILTIADIS
LICENSE: MIT LICENSE
'''

import mplfinance as mpf

# Add the parent directory to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.ticker_utils import get_ticker


# We go as low as 15 min charts to reduce cluttering (start of day on 1-5 min is very volatile)
#To find the support and resistance levels, we will use the rolling minimum and maximum functions in pandas.
#The rolling minimum and maximum functions will calculate the minimum and maximum values over a specified window size.
def support_resistance(data, window=20):
    support = data['Low'].rolling(window=window).min()
    resistance = data['High'].rolling(window=window).max()
    return support, resistance


# Minimum frequency of support and resistance levels to be considered as major levels (i.e. how many times the value was reached)
def find_major_levels(data,min_freq_s=2, min_freq_r=2,window=20):
    support, resistance = support_resistance(data, window)
    data['Support'] = support
    data['Resistance'] = resistance
    print(data.head())

    ms = data.Support.value_counts()
    mr = data.Resistance.value_counts()
    print('Support:', ms)
    print('Resistance:', mr)
    major_support =  ms[ms >= min_freq_s].keys()
    major_resistance = mr[mr >= min_freq_r].keys()
    print('Major Support:', major_support)
    print('Major Resistance:', major_resistance)
    #TODO: Define algorithm to reduce cluttering on values (i.e. if 2 values are very close to each other, remove one)
    
    return major_support,major_resistance,support,resistance

# Plot the support and resistance levels on a candlestick chart using the mplfinance library.
def plot_support_resistance(data, major_support, major_resistance):
    sp = []
    for s in major_support:
        #TODO: Define algorithm to reduce cluttering on the plot
        sp.append(s)
    res = []
    for r in major_resistance:
        res.append(r)    

# hlines is used to plot horizontal lines on the chart for the support and resistance levels.
    mpf.plot(data, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', hlines=dict(hlines=sp+res, colors=['b'], linestyle='--'))


def find_support_resistance_levels(stock_symbol, timespan, time, window=20, min_freq_s=2, min_freq_r=2):
    data = get_ticker(stock_symbol, timespan, time)
    major_support, major_resistance,support,resistance = find_major_levels(data,min_freq_s,min_freq_r,window)
    return data,major_support, major_resistance

def example():
    df_daily,major_support, major_resistance = find_support_resistance_levels('AAPL', '5d', '15m')
    plot_support_resistance(df_daily, major_support, major_resistance)

'''
# Split data based on date and time
# 4y data to find the all_time support and resistance levels.
# 1y data to find the yearly support and resistance levels.
# 1mo data to find the monthly support and resistance levels.
# 5d data to find the weekly support and resistance levels.
# 1d data to find the daily support and resistance levels.

df_daily = get_ticker('AAPL', '1d', '15m')      #Daily Data
df_weekly = get_ticker('AAPL', '5d', '15m')     #Weekly Data
df_monthly = get_ticker('AAPL', '1mo', '1d')    #Monthly Data
df_yearly = get_ticker('AAPL', '1y', '1d')      #Yearly Data
df_4y = get_ticker('AAPL', '5y', '1d')          #4y Data (max)

major_support, major_resistance,support,resistance = find_major_levels(df_daily)
plot_support_resistance(df_daily, major_support, major_resistance)
'''
