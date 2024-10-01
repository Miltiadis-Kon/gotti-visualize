# Import the libraries
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Load the data 
from data.ticker_utils import get_ticker


from get_pivot_points import getPivot
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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

# Plot the pivot points and the data on a candlestick chart
def plot_support_resistance_pivotpoints(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                hovertext=data['date'].dt.strftime('%d/%m/%Y %H:%M'),
                increasing_line_color= 'green', 
                decreasing_line_color= 'red'),        
])
# Add the pivot points as markers
    fig.add_scatter(x=data.index, y=data['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")
# Draw a straight line for each pivot point
    for i in range(len(data)):
        if not np.isnan(data['pointpos'][i]):
            if data['pivot'][i] == 1:
                fig.add_shape(type="line",
                      x0=0, y0=data['pointpos'][i],
                      x1=data.index[-1], y1=data['pointpos'][i],
                      line=dict(color="olive", width=1))
            else:
                fig.add_shape(type="line",
                      x0=0, y0=data['pointpos'][i],
                      x1=data.index[-1], y1=data['pointpos'][i],
                      line=dict(color="maroon", width=1))
                                 
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title='Support Resistance with Pivot Points')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    fig.show()


def find_pivot_points(stock_symbol, timespan, time):
    data = get_ticker(stock_symbol, timespan, time)
    data = getPivot(data,time)
    # Calculate the pivot points for each candle in the dataframe and store the result in a new column called 'pivot'
    # Calculate the position of the pivot points based on the pivot type (pivot low or pivot high)
    # Return the dataframe with the pivot points > 0 (pivot low or pivot high)
    support = data[data['pivot']==1]
    resistance = data[data['pivot']==2]
    plot_support_resistance_pivotpoints(data)
    return data, support['pointpos'].dropna().values, resistance['pointpos'].dropna().values

def histogram(data,bandwidth=3):
    # Filter the dataframe based on the pivot column
    high_values = data[data['pivot'] == 2]['high']
    low_values = data[data['pivot'] == 1]['low']
    # Define the bin width
    # Calculate the number of bins
    bins = int((high_values.max() - low_values.min()) / bandwidth)
    # Create the histograms
    plt.figure(figsize=(10, 5))
    plt.hist(high_values, bins=bins, alpha=0.5, label='High Values', color='red')
    plt.hist(low_values, bins=bins, alpha=0.5, label='Low Values', color='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of High and Low Values')
    plt.legend()
    plt.show()
    
    

find_pivot_points('AAPL', '1mo', '30m')
