import numpy as np
import pandas as pd
from collections import defaultdict

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.ticker_utils import get_ticker


def isPivot(dataframe, candle_index, window):
    '''
    Check x candles on the left side and y candles on the right side of current candle.
    If this candle is the highest among x previous and y afterwards then FLAG it as high
    If it is lower among x and y then FLAG it as low
    '''
    # Ensure the index is an integer
    if isinstance(candle_index, pd.Timestamp):
        candle_index = dataframe.index.get_loc(candle_index)
    # Check if the index is within the range of the dataframe
    if candle_index - window < 0 or candle_index + window >= len(
        dataframe
    ):
        return 0

    pivotlow = 1
    pivothigh = 1
    # Check if the low price of the current candle is the lowest in the range of candles to the left and right
    for i in range(candle_index - window, candle_index + window + 1):
        if dataframe["low"].iloc[candle_index] > dataframe["low"].iloc[i]:
            pivotlow = 0
        if dataframe["high"].iloc[candle_index] < dataframe["high"].iloc[i]:
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

def pointpos(x, spread=0):
    '''
    Increment or decrement (spread) the value of the pivot point for better support/resistance identification
    '''
    # If the pivot point is a pivot low, decrement the low price by a small value to avoid overlap with the candle
    if x["pivot"] == 1:
        return x["low"] # - spread
    # If the pivot point is a pivot high, increment the high price by a small value to avoid overlap with the candle
    elif x["pivot"] == 2:
        return x["high"] # + spread
    else:
        # If the pivot point is not a pivot point, return NaN
        return np.nan

#TODO : FIX THIS - Optimize 
def optimize(time):
    """
    A dictionary to store the optimized parameters for each timespan and time
    on small timeframes, we have low number of candles, so we need to increase the range
    on large timeframes, we have high number of candles, so we need to decrease the range
    on small timeframes, we have high volatility, so we need to increase the spread
    on large timeframes, we have low volatility, so we need to decrease the spread
    """
    # Define the default parameters for the pivot points
    if time == '15m':
        return 10
    elif time == '1h':
        return 5
    elif time =='30m':
        return 5

def getPivot(data, time):
    window = optimize(time)
    data['pivot'] = data.apply(lambda x: isPivot(data, x.name, window), axis=1)
    '''
    OLD METHOD
    TODO:REDUCE IT DOWN TO A SINGLE ITERATION
    lows=[]
    highs=[]
    for i in range(0,len(data)):
        if data['pivot'][i]==1:
            lows.append(data['low'][i])
        elif data['pivot'][i]==2:
            highs.append(data['high'][i])          
    optimizePivot(highs)
    optimizePivot(lows)    
    '''
    tolerance = 0 # 5% tolerance
    #data = update_pivot_values(data,'high',tolerance)
    data = update_pivot_values(data,tolerance)
    data['pointpos'] = data.apply(lambda row: pointpos(row), axis=1)
    return data


#TODO: FIX this
def update_pivot_values(df, tolerance=0.1):
    # Function to find values within tolerance range
    def find_within_range(series, value, tolerance):
        lower_bound = value * (1 - tolerance)
        upper_bound = value * (1 + tolerance)
        return (series >= lower_bound) & (series <= upper_bound)

    # Update pivot = 1 values
    pivot_1_mask = df['pivot'] == 1
    pivot_1_values = df.loc[pivot_1_mask, 'low']
    
    for idx, value in pivot_1_values.items():
        within_range = find_within_range(df.loc[pivot_1_mask, 'low'], value, tolerance)
        df.loc[pivot_1_mask & within_range & (df.index != idx), 'pivot'] = 0

    # Update pivot = 2 values
    pivot_2_mask = df['pivot'] == 2
    pivot_2_values = df.loc[pivot_2_mask, 'high']
    
    for idx, value in pivot_2_values.items():
        within_range = find_within_range(df.loc[pivot_2_mask, 'high'], value, tolerance)
        df.loc[pivot_2_mask & within_range & (df.index != idx), 'pivot'] = 0

    return df

'''
Each trading session is 6:30 hours = 390 minutes
From 16:30 - 23:00 (GMT+2) or 9:30-16:00 (ET)

390 minutes / day is 

7 candles of 1h -   for major s/r a window of 6 (6L 6R) is good given that data are  (1M < data < 1Y)
13 candles of 30min - a window of 6 is also good here for (1M < data < 5D)
26 candles of 15 min - a window of 12 is good 
'''
