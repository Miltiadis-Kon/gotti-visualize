"""
Levels Visualizer

Interactive Streamlit app to visualize support/resistance levels calculated during backtest.
Features:
- Load levels history JSON from backtest
- Date slider to see levels for each day
- Chart with price data and horizontal lines for supports/resistances
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.key_levels import KeyLevels, TIMEFRAME_LOOKBACK


# Page config
st.set_page_config(
    page_title="Levels Visualizer",
    page_icon="📊",
    layout="wide"
)

# Custom dark theme
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stSelectbox > div > div { background-color: #1e2130; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Daily Levels Visualizer")

# Sidebar - Load levels file
st.sidebar.header("📁 Load Levels Data")

logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Find levels files
levels_files = []
if os.path.exists(logs_dir):
    for f in os.listdir(logs_dir):
        if f.endswith('_levels.json'):
            levels_files.append(os.path.join(logs_dir, f))

levels_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

if not levels_files:
    st.warning("No levels files found. Run a backtest first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select Levels File",
    levels_files,
    format_func=lambda x: os.path.basename(x)
)

# Load levels data
@st.cache_data
def load_levels_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

try:
    data = load_levels_data(selected_file)
    ticker = data['ticker']
    strategy = data['strategy']
    levels_list = data['levels']
    
    st.sidebar.success(f"✅ Loaded {len(levels_list)} days for {ticker}")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Date selection
st.sidebar.header("📅 Select Date")

dates = [entry['date'] for entry in levels_list]
start_date = dates[0]  # First date in backtest

date_index = st.sidebar.slider(
    "Day",
    0, len(dates) - 1,
    value=len(dates) - 1,
    format=f"Day %d"
)

selected_date = dates[date_index]
selected_levels = levels_list[date_index]

st.sidebar.markdown(f"**Start Date:** {start_date}")
st.sidebar.markdown(f"**Selected Date:** {selected_date}")
st.sidebar.markdown(f"**Supports:** {len(selected_levels['supports'])}")
st.sidebar.markdown(f"**Resistances:** {len(selected_levels['resistances'])}")

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Chart Timeframe",
    ['5m', '15m', '1h', '4h', '1d'],
    index=0
)

# Fetch price data for a single day
@st.cache_data(ttl=3600)
def get_price_data_single_day(ticker, date_str, interval='5m'):
    """Fetch OHLCV data for a single day."""
    try:
        import yfinance as yf
        
        start_dt = datetime.strptime(date_str, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=1)  # Just one day
        
        stock = yf.Ticker(ticker)
        df = stock.history(
            start=start_dt,
            end=end_dt,
            interval=interval,
            prepost=True
        )
        
        # Lowercase columns
        df.columns = [col.lower() for col in df.columns]
        df = df[df['volume'] != 0]
        df.reset_index(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return None

# Get price data for selected day only
df = get_price_data_single_day(ticker, selected_date, timeframe)

if df is None or df.empty:
    st.warning("No price data available")
    st.stop()

# Create chart
fig = go.Figure()

# Find the date column - Yahoo Finance uses 'Date' or 'Datetime' after reset_index
date_col = None
for col in df.columns:
    if col.lower() in ['date', 'datetime', 'timestamp', 'index']:
        date_col = col
        break

if date_col:
    x_dates = pd.to_datetime(df[date_col])
else:
    x_dates = df.index

# Add candlestick
fig.add_trace(go.Candlestick(
    x=x_dates,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='Price',
    increasing=dict(line=dict(color='#26a69a')),
    decreasing=dict(line=dict(color='#ef5350'))
))

# Add support levels (green)
for support in selected_levels['supports']:
    fig.add_hline(
        y=support['price'],
        line=dict(color='#00ff00', width=1 + support['importance'] * 0.5, dash='solid'),
        annotation=dict(
            text=f"S ${support['price']:.2f} (imp={support['importance']})",
            font=dict(color='#00ff00', size=10),
            bgcolor='rgba(0,0,0,0.7)'
        ),
        annotation_position="left"
    )

# Add resistance levels (red)
for resistance in selected_levels['resistances']:
    fig.add_hline(
        y=resistance['price'],
        line=dict(color='#ff4444', width=1 + resistance['importance'] * 0.5, dash='solid'),
        annotation=dict(
            text=f"R ${resistance['price']:.2f} (imp={resistance['importance']})",
            font=dict(color='#ff4444', size=10),
            bgcolor='rgba(0,0,0,0.7)'
        ),
        annotation_position="right"
    )

# Mark selected date (handle both DatetimeIndex and regular index)
try:
    if hasattr(df.index, 'strftime'):
        date_list = df.index.strftime('%Y-%m-%d').tolist()
    elif 'date' in df.columns:
        date_list = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').tolist()
    else:
        date_list = []
    
    if selected_date in date_list:
        fig.add_vline(
            x=selected_date,
            line=dict(color='yellow', width=2, dash='dash'),
            annotation=dict(text="Selected Date", font=dict(color='yellow'))
        )
except Exception:
    pass  # Skip date marking if it fails

# Layout
# Calculate Y-axis range based on price data (with 5% padding)
price_min = df['low'].min()
price_max = df['high'].max()
price_range = price_max - price_min
y_min = price_min - price_range * 0.05
y_max = price_max + price_range * 0.05

fig.update_layout(
    title=f"{ticker} - Levels on {selected_date}",
    template='plotly_dark',
    height=700,
    xaxis_rangeslider_visible=False,
    showlegend=False,
    plot_bgcolor='#0e1117',
    paper_bgcolor='#0e1117',
    xaxis=dict(
        gridcolor='rgba(128,128,128,0.1)',
        showgrid=True,
        # Remove gaps for non-trading hours
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[20, 4], pattern="hour"),  # Hide overnight (8pm to 4am)
        ]
    ),
    yaxis=dict(
        gridcolor='rgba(128,128,128,0.1)',
        showgrid=True,
        side='right',
        range=[y_min, y_max]  # Auto-scale to price range
    )
)

# Display chart
st.plotly_chart(fig, use_container_width=True)

# Show levels table
col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Support Levels")
    if selected_levels['supports']:
        supports_df = pd.DataFrame(selected_levels['supports'])
        supports_df = supports_df.sort_values('price', ascending=False)
        st.dataframe(supports_df, use_container_width=True)
    else:
        st.info("No supports found")

with col2:
    st.subheader("🔴 Resistance Levels")
    if selected_levels['resistances']:
        resistances_df = pd.DataFrame(selected_levels['resistances'])
        resistances_df = resistances_df.sort_values('price', ascending=True)
        st.dataframe(resistances_df, use_container_width=True)
    else:
        st.info("No resistances found")

# Footer
st.markdown("---")
st.markdown(f"**Strategy:** {strategy} | **Ticker:** {ticker} | **Days:** {len(levels_list)}")
