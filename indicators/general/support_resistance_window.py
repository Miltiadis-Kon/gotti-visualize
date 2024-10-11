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
import plotly.graph_objects as go

# Add the parent directory to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.ticker_utils import get_ticker


# We go as low as 15 min charts to reduce cluttering (start of day on 1-5 min is very volatile)
#To find the support and resistance levels, we will use the rolling minimum and maximum functions in pandas.
#The rolling minimum and maximum functions will calculate the minimum and maximum values over a specified window size.
def support_resistance(data, window=20):
    support = data['low'].rolling(window=window).min()
    resistance = data['high'].rolling(window=window).max()
    return support, resistance


# Minimum frequency of support and resistance levels to be considered as major levels (i.e. how many times the value was reached)
def find_major_levels(data,min_freq_s=2, min_freq_r=2,window=20):
    try:
        support, resistance = support_resistance(data, window)
        data['support'] = support
        data['resistance'] = resistance
        #print(data.head())
    except:
        print('Error calculating support and resistance levels. Window size might be too large/too small')
        return None,None,None,None
    
    ms = data.support.value_counts()
    mr = data.resistance.value_counts()
    #print('Support:', ms)
    #print('Resistance:', mr)
    try:
        major_support =  ms[ms >= min_freq_s].keys()
        major_resistance = mr[mr >= min_freq_r].keys()
        #print('Major Support:', major_support)
        #print('Major Resistance:', major_resistance)
    except:
        print('Error finding major support and resistance levels. Frequency might be too high or data might be too small')
        return None,None,None,None
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

# Plot the support and resistance levels on a candlestick chart using the Plotly library.
def plot_support_resistance_ploty(data, major_support, major_resistance):
    fig = go.Figure()
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Candlesticks',
        hovertext=data['date'].dt.strftime('%d/%m/%Y %H:%M')
    ))
    # Add support lines
    for s in major_support:
        # TODO: Define algorithm to reduce cluttering on the plot
        fig.add_hline(y=s, line=dict(color='green', dash='dash'), name='Support')

    # Add resistance lines
    for r in major_resistance:
        fig.add_hline(y=r, line=dict(color='red', dash='dash'), name='Resistance')

    # Update layout
    fig.update_layout(
        title='Support and Resistance with Rolling Window',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
    fig.show()

def find_support_resistance_levels(stock_symbol, timespan, time, window=40, min_freq_s=4, min_freq_r=8):
    try:
        data = get_ticker(stock_symbol, timespan, time)
    except:
        print('Error fetching data')
        return None,None,None
    major_support, major_resistance,support,resistance = find_major_levels(data,min_freq_s,min_freq_r,window)
    return data,major_support, major_resistance




def example():
    df_daily,major_support, major_resistance = find_support_resistance_levels('AAPL', '5d', '15m')
    #plot_support_resistance(df_daily, major_support, major_resistance)
    plot_support_resistance_ploty(df_daily, major_support, major_resistance)

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
