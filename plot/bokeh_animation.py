from bokeh.models import Button, Slider, DatetimeTickFormatter
from bokeh.plotting import figure, column, row
from bokeh.models.ranges import FactorRange
from bokeh.io import curdoc

import time
import pandas as pd

## Prepare Data
df = pd.read_csv("data/AAPL_stock_data_1d_1y.csv")
df.rename(columns={'Datetime': 'Date'}, inplace=True)
if '-04:00' in df['Date'][0]:
    df['Date'] = df['Date'].str.slice(0, -6)
    
df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d %H:%M:%S')  # Adjust this
df.columns = df.columns.str.lower()
df["date"] = pd.to_datetime(df["date"])

## Create Candlestick Charts
days = 22*3 ## 3 Months

inc = df[:days].close > df[:days].open
dec = df[:days].open > df[:days].close

w = 12*60*60*1000

fig = figure(x_axis_type="datetime", width=900, height=500,
             #y_range=(20,120),
             title = "Microsoft Candlestick Chart Animation (2000-2013)")

segments = fig.segment("date", "high", "date", "low", color="black", source=df[:days])

green_patterns = fig.vbar("date", w, "open", "close", fill_color="lawngreen", line_width=0,
                          source=df[:days][inc])

red_patterns = fig.vbar("date", w, "open", "close", fill_color="tomato", line_width=0,
                        source=df[:days][dec])

fig.xaxis.axis_label="Date"
fig.yaxis.axis_label="Price ($)"

fig.xaxis.formatter = DatetimeTickFormatter(days="%m-%d-%Y")

## Define Widgets
btn = Button(label="Play")

## Define Callbacks
curr_cnt = days
def update_chart():
    global curr_cnt
    curr_cnt += 1
    if curr_cnt == len(df):
        curr_cnt = days

    inc = df[curr_cnt-days:curr_cnt].close > df[curr_cnt-days:curr_cnt].open
    dec = df[curr_cnt-days:curr_cnt].open > df[curr_cnt-days:curr_cnt].close

    segments.data_source.data = df[curr_cnt-days:curr_cnt]
    green_patterns.data_source.data = df[curr_cnt-days:curr_cnt][inc]
    red_patterns.data_source.data = df[curr_cnt-days:curr_cnt][dec]

callback = None
def execute_animation():
    global callback
    if btn.label == "Play":
        btn.label = "Pause"
        callback = curdoc().add_periodic_callback(update_chart, 200)
    else:
        btn.label = "Play"
        curdoc().remove_periodic_callback(callback)


## Register Callbacks
btn.on_click(execute_animation)

## GUI
curdoc().add_root(column(btn, fig))