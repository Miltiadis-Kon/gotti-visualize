import plotly.graph_objects as go
import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

data = pd.read_csv('data/AAPL_stock_data_1h_6mo.csv')

# Some data processing
#Rename headers to lowercase
data.columns = data.columns.str.lower()
# Add datetime as a column
data.reset_index(drop=True, inplace=True)



play_button = {
    "args": [
        None, 
        {
            "frame": {"duration": 300, "redraw": True},
            "fromcurrent": True, 
            "transition": {"duration": 100,"easing": "quadratic-in-out"}
        }
    ],
    "label": "Play",
    "method": "animate"
}

pause_button = {
    "args": [
        [None], 
        {
            "frame": {"duration": 0, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 0}
        }
    ],
    "label": "Pause",
    "method": "animate"
}

sliders_steps = [
    {"args": [
        [0, i], # 0, in order to reset the image, i in order to plot frame i
        {"frame": {"duration": 300, "redraw": True},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
    "label": i,
    "method": "animate"}
    for i in range(len(data))      
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "candle:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": sliders_steps,
}

initial_plot = go.Candlestick(
    x=data.index, 
    open=data.open, 
    high=data.high, 
    low=data.low, 
    close=data.close
)

first_i_candles = lambda i: go.Candlestick(
    x=data.index, 
    open=data.open[:i], 
    high=data.high[:i], 
    low=data.low[:i], 
    close=data.close[:i]
)

fig = go.Figure(
    data=[initial_plot],
    layout=go.Layout(
        xaxis=dict(title='Date', rangeslider=dict(visible=False)),
        title="Candles over time",
        updatemenus= [dict(type="buttons", buttons=[play_button, pause_button])],
        sliders = [sliders_dict]
    ),
    frames=[
        go.Frame(data=[first_i_candles(i)], name=f"{i}") # name, I imagine, is used to bind to frame i :) 
        for i in range(len(data))
    ]
)

fig.show()