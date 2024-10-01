'''
Price breakout & retest signal indicator
If the price tests a support/resistance level and breaks out, the indicator will return a breakout signal
Then it will wait for the price to retest the support/resistance level and return a signal
The second signal will be a strong indiator of a position placement.
NOTE: To reduce outdated signals i.e. price breakout after a long time, 
the indicator will have a FIXED number of candles to wait for the retest signal.
TODO: Implement a dynamic number of candles to wait for the retest signal
'''
from support_resistance_pivotpoints import pointpos, isPivot
import plotly.graph_objects as go

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.ticker_utils import get_ticker

# Indicates that the price has broken out of a support/resistance level
def breakout_signal(dataframe,candle, backcandles, window,spread=0.002):
    """
    NOTE: dataframe must have a column 'isPivot' with values 1,2,3,0
    Attention! window should always be greater than the pivot window! to avoid look ahead bias
    """
    if (candle <= (backcandles+window)) or (candle+window+1 >= len(dataframe)):
        return 0 
    localdf = dataframe.iloc[candle-backcandles-window:candle-window] #window must be greater than pivot window to avoid look ahead bias
    highs = localdf[localdf['pivot'] == 1].high.tail(3).values
    lows = localdf[localdf['pivot'] == 2].low.tail(3).values
    levelbreak = 0
    if len(lows)==3:
        support_condition = True # Consecutive lows detected! 
        mean_low = lows.mean()
        for low in lows:
            if abs(low-mean_low)>spread:
                support_condition = False # If the lows are not aligned (i.e. no breakdown just a downtrend)
                break
        if support_condition and (mean_low - dataframe.loc[candle].close)>spread*2: # If alligned correctly
            levelbreak = 1 # We have a support breakout signal - Potential short

    if len(highs)==3:
        resistance_condition = True # Consecutive highs detected!
        mean_high = highs.mean()
        for high in highs:
            if abs(high-mean_high)>spread:
                resistance_condition = False # If the highs are not aligned (i.e. no breakout just an uptrend)
                break
        if resistance_condition and (dataframe.loc[candle].close-mean_high)>spread*2: # If alligned correctly
            levelbreak = 2 # We have a resistance breakout signal - Potential long
    return levelbreak

# Indicates that the price has retested a support/resistance level after a breakout
#TODO:FIX THIS
def retest_signal(dataframe, candle, backcandles, window,spread=0.002):
    '''
    Retest signal checks if the price retests the support/resistance level after a breakout
    NOTE: dataframe must have a column 'isPivot' with values 1,2,3,0
    Attention! window should always be greater than the pivot window! to avoid look ahead bias
    '''
    if (candle <= (backcandles+window)) or (candle+window+1 >= len(dataframe)):
        return 0 
    # Check x previous candles for a breakout signal
    retest = 0
    localdf = dataframe.iloc[candle-backcandles-window:candle-window] #window must be greater than pivot window to avoid look ahead bias
    for i in range(len(localdf)):
        if localdf['breakout_signal'][i] == 1: # Support breakout signal
            if abs(dataframe.loc[candle].low - localdf['low'][i]) < spread : # Retest of the support level previous candle ( price below the low of the breakout candle)
                retest=1 # if price retests and candle closes above the retest
        elif localdf['breakout_signal'][i] == 2: # Resistance breakout signal
            if abs(dataframe.loc[candle].high - localdf['high'][i]) < spread : # Retest of the resistance level previous candle ( price above the high of the breakout candle)
                retest =1 # if price retests and candle closes below the retest
    return retest

# Indicates that the price has broken out of a support/resistance level after a retest
#TODO:FIX THIS
def retest_entry (dataframe, candle, backcandles, window,spread=0.002):
    '''
    Retest signal checks if the price retests the support/resistance level after a breakout
    NOTE: dataframe must have a column 'isPivot' with values 1,2,3,0
    Attention! window should always be greater than the pivot window! to avoid look ahead bias
    '''
    if (candle <= (backcandles+window)) or (candle+window+1 >= len(dataframe)):
        return 0 
    # Check x previous candles for a breakout signal
    entry = 0
    localdf = dataframe.iloc[candle-backcandles-window:candle-window] #window must be greater than pivot window to avoid look ahead bias
    for i in range(len(localdf)):
        if localdf['retest_signal'][i] == 1: # Support retest/breakout signal
            if dataframe.loc[candle].close < localdf['close'][i] : # Retest of the support level previous candle ( price below the low of the breakout candle)
                entry=1 # Go short
        elif localdf['retest_signal'][i] == 2: # Resistance retest/breakout signal
            if dataframe.loc[candle].close > localdf['close'][i]: # Retest of the resistance level previous candle ( price above the high of the breakout candle)
                entry =2 # Go long
    return entry
        

def detect_breakout(data):
    #Check if pivot column exists
    spread = 0.002*data['close'].mean()
    if 'pivot' not in data.columns:
        print("Pivot column not found in the dataframe! Recalculation in progress...!")
        print("Using default parameters: left_candles=15, right_candles=15 spread=0.2% mean val")
        spread = 0.002*data['close'].mean()
        left_candles = 15
        right_candles = 15
        data['pivot'] = data.apply(lambda x: isPivot(data, x.name, left_candles, right_candles), axis=1)
        # Calculate the position of the pivot points based on the pivot type (pivot low or pivot high)
        data['pointpos'] = data.apply(lambda row: pointpos(row,spread), axis=1)
    
    #TODO: Implement a dynamic algo for window size,spread and backcandles
    #Calculate the breakout signal    
    data['breakout_signal'] = data.apply(lambda x: breakout_signal(data, x.name, 60, 20,spread), axis=1)
    data['retest_signal'] = data.apply(lambda x: retest_signal(data, x.name, 40, 20,spread), axis=1)
    data['retest_entry'] = data.apply(lambda x: retest_entry(data, x.name, 3, 20,spread), axis=1) # MAX 3 CANDLES AFTER RETEST
    return data

def plot_breakout_signal(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])
    # Add breakout signals to the plot
    for i in range(len(data)):
        if data['breakout_signal'][i] == 1:
            fig.add_annotation(x=data.index[i], y=data['low'][i], text="Support Breakout", showarrow=True, arrowhead=1)
        if data['retest_signal'][i] == 1:
            fig.add_annotation(x=data.index[i], y=data['low'][i], text="Support Retest", showarrow=True, arrowhead=1)
        if data['retest_entry'][i] == 1:
            fig.add_annotation(x=data.index[i], y=data['low'][i], text="Short Entry", showarrow=True, arrowhead=1)
        if data['retest_enrty'][i] == 2:
            fig.add_annotation(x=data.index[i], y=data['high'][i], text="Long Entry", showarrow=True, arrowhead=1)
        if data['breakout_signal'][i] == 2:
            fig.add_annotation(x=data.index[i], y=data['high'][i], text="Resistance Breakout", showarrow=True, arrowhead=1)   
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title='Breakout Signal')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
    fig.show()
    print(data[data['breakout_signal']!=0])


# Example usage
data = get_ticker('AAPL', '1mo', '15m')
detect_breakout(data)
plot_breakout_signal(data)