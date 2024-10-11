import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Example usage
data = pd.read_csv('data\AAPL_stock_data_1h_6mo.csv')
#Rename headers to lowercase
data.columns = data.columns.str.lower()
# Add datetime as a column
data.reset_index(drop=True, inplace=True)

'''
# Detect all doji stars
doji_pattern = data.ta.cdl_pattern(name="doji")
doji_pattern.rename(columns={'CDL_DOJI_10_0.1': 'doji_star'}, inplace=True)
data = data.join(doji_pattern)
data['date'] = data.index
'''

# Detect all candlestick patterns
candle_patterns = data.ta.cdl_pattern(name="all")
print(candle_patterns.columns)


# Rename the column to make it easier to access

# Plot candlestick chart with doji stars

'''
fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])
for i in range(len(data)):
        if data['doji_star'].iloc[i] >90.0:     
            fig.add_annotation(x=data.index[i], y=data['low'][i], text="Doji", showarrow=True, arrowhead=1)
        if data['morning_star'].iloc[i] >90.0:
            fig.add_annotation(x=data.index[i], y=data['low'][i], text="Morning Star", showarrow=True, arrowhead=1)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(title='Doji Signal')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
'''
