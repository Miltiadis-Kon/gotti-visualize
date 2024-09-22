'''
Here we utilize pivot points to determine support and resistance levels.
Pivot points are calculated as follows:


'''
# Import the libraries
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Load the data 
from data.ticker_utils import get_ticker

'''
Calculate the pivot points based on the high, low, and close prices
Method to calculate pivot points:
    1. Compare candle price with 2n neighbours to the left and right of the candle.
    2. Return the following values:
        1 if the low price of the current candle is the lowest in the range of candles to the left and right
        2 if the high price of the current candle is the highest in the range of candles to the left and right
        3 if the current candle has both pivotlow and pivothigh
        0 if the current candle is not a pivot point 
    3. Increment or decrement the value of the pivot point for better support/resistance identification
'''
def pivotid(dataframe, candle_index, left_candles, right_candles): 
        # Ensure the index is an integer
    if isinstance(candle_index, pd.Timestamp):
        candle_index = dataframe.index.get_loc(candle_index)
    # Check if the index is within the range of the dataframe
    if candle_index-left_candles < 0 or candle_index+right_candles >= len(dataframe):
        return 0
    
    pivotlow=1
    pivothigh=1
    # Check if the low price of the current candle is the lowest in the range of candles to the left and right
    for i in range(candle_index - left_candles, candle_index + right_candles + 1):
        if dataframe['low'].iloc[candle_index] > dataframe['low'].iloc[i]:
            pivotlow = 0
        if dataframe['high'].iloc[candle_index] < dataframe['high'].iloc[i]:
            pivothigh = 0
            
    if pivotlow and pivothigh:
        # Return both if candle has both pivotlow and pivothigh (rare, usually means a doji or high tail candle)
        return 3
    elif pivotlow:
        # Return pivotlow if the low price of the current candle is the lowest in the range of candles to the left and right
        return 1
    elif pivothigh:
        # Return pivothigh if the high price of the current candle is the highest in the range of candles to the left and right
        return 2
    else:
        # Return 0 if the current candle is not a pivot point
        return 0

# Increment or decrement the value of the pivot point for better support/resistance identification
def pointpos(x):
    # If the pivot point is a pivot low, decrement the low price by a small value to avoid overlap with the candle
    if x['pivot']==1:
        return x['low']-1e-3
    # If the pivot point is a pivot high, increment the high price by a small value to avoid overlap with the candle
    elif x['pivot']==2:
        return x['high']+1e-3
    else:
        # If the pivot point is not a pivot point, return NaN
        return np.nan

# Plot the pivot points and the data on a candlestick chart
def plot_support_resistance_pivotpoints(data):
    dfpl = data
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['open'],
                high=dfpl['high'],
                low=dfpl['low'],
                close=dfpl['close'],
                increasing_line_color= 'green', 
                decreasing_line_color= 'red')])
# Add the pivot points as markers
    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")
# Draw a straight line for each pivot point
    for i in range(len(dfpl)):
        if not np.isnan(dfpl['pointpos'][i]):
            fig.add_shape(type="line",
                      x0=dfpl.index[i], y0=dfpl['pointpos'][i],
                      x1=dfpl.index[-1], y1=dfpl['pointpos'][i],
                      line=dict(color="MediumPurple", width=1))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title='Support Resistance with Pivot Points')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
    fig.show()


def find_pivot_points(stock_symbol, timespan, time, left_candles, right_candles):
    data = get_ticker(stock_symbol, timespan, time)
    # Calculate the pivot points for each candle in the dataframe and store the result in a new column called 'pivot'
    data['pivot'] = data.apply(lambda x: pivotid(data, x.name, left_candles, right_candles), axis=1)
    # Calculate the position of the pivot points based on the pivot type (pivot low or pivot high)
    data['pointpos'] = data.apply(lambda row: pointpos(row), axis=1)
    # Return the dataframe with the pivot points > 0 (pivot low or pivot high)
    support = data[data['pivot']==1]
    resistance = data[data['pivot']==2]
    return data, support['pointpos'].dropna().values, resistance['pointpos'].dropna().values
