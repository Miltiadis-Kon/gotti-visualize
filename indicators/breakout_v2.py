import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


# Import the libraries
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Load the data 
from data.ticker_utils import get_ticker
from get_pivot_points import isPivot,pointpos,optimize

data = get_ticker('AAPL','5d','15m')

def get_pivot(data):
    window,spread = optimize(data,'5d', '15m') # optional
    # Calculate the pivot points for each candle in the dataframe and store the result in a new column called 'pivot'
    data['pivot'] = data.apply(lambda x: isPivot(data, x.name, window), axis=1)
    # Calculate the position of the pivot points based on the pivot type (pivot low or pivot high)
    data['pointpos'] = data.apply(lambda row: pointpos(row), axis=1)

def collect_channel(data):
    highs = data[data['pivot']==1].high.values
    idxhighs = data[data['pivot']==1].high.index
    lows = data[data['pivot']==2].low.values
    idxlows = data[data['pivot']==2].low.index
    
    if len(lows)>=2 and len(highs)>=2:
        sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows,lows)
        sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs,highs)
    
        return(sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return(0,0,0,0,0,0)

def plot(data,candle, backcandles, window):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])

    fig.add_scatter(x=data.index, y=data['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")

    fig.add_scatter(x=data.index, y=data['breakpointpos'], mode="markers",
                marker=dict(size=8, color="Black"), marker_symbol="triangle-down",
                name="breakdown")

    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(data)
    print(r_sq_l, r_sq_h)
    x = np.array(range(candle-backcandles-window, candle+1))
    fig.add_trace(go.Scatter(x=x, y=sl_lows*x + interc_lows, mode='lines', name='lower slope'))
    fig.add_trace(go.Scatter(x=x, y=sl_highs*x + interc_highs, mode='lines', name='max slope'))
    #fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


def isBreakOut(df,candle, backcandles, window):
    if (candle-backcandles-window)<0:
        return 0
    
    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(df)
    
    prev_idx = candle-1
    prev_high = df.iloc[candle-1].high
    prev_low = df.iloc[candle-1].low
    prev_close = df.iloc[candle-1].close
    
    curr_idx = candle
    curr_high = df.iloc[candle].high
    curr_low = df.iloc[candle].low
    curr_close = df.iloc[candle].close
    curr_open = df.iloc[candle].open

    if ( prev_high > (sl_lows*prev_idx + interc_lows) and
        prev_close < (sl_lows*prev_idx + interc_lows) and
        curr_open < (sl_lows*curr_idx + interc_lows) and
        curr_close < (sl_lows*prev_idx + interc_lows)): #and r_sq_l > 0.9
        return 1
    
    elif ( prev_low < (sl_highs*prev_idx + interc_highs) and
        prev_close > (sl_highs*prev_idx + interc_highs) and
        curr_open > (sl_highs*curr_idx + interc_highs) and
        curr_close > (sl_highs*prev_idx + interc_highs)): #and r_sq_h > 0.9
        return 2
    
    else:
        return 0

def breakpointpos(x):
    if x['isBreakOut']==2:
        return x['low']-3e-3
    elif x['isBreakOut']==1:
        return x['high']+3e-3
    else:
        return np.nan


candle = 75
backcandles = 40
window = 3


get_pivot(data)
print(data.head())
data["isBreakOut"] = [isBreakOut(data,candle, backcandles, window) for candle in data.index]
data['breakpointpos'] = data.apply(lambda row: breakpointpos(row), axis=1)
plot(data,candle, backcandles, window)
