"""
Key Levels Dashboard - Interactive TradingView-like Charting

Features:
- Ticker selection
- Timeframe dropdown (1D, 4H, 1H, 15m, 5m)
- Toggle visibility for key levels and Fibonacci
- Filter by importance level
- Interactive Plotly chart with zoom/pan

Run with: streamlit run plots/key_levels_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.key_levels import (
    KeyLevels, 
    TIMEFRAME_COLORS, 
    TIMEFRAME_LOOKBACK,
    TIMEFRAME_IMPORTANCE,
    FIBONACCI_LEVELS,
    FIBONACCI_THRESHOLD,
    FIBONACCI_IMPORTANCE,
    PRICE_THRESHOLD
)

# Page config
st.set_page_config(
    page_title="Key Levels Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for TradingView-like dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #131722;
    }
    .stSidebar {
        background-color: #1e222d;
    }
    h1, h2, h3, p, label {
        color: #d1d4dc !important;
    }
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #d1d4dc !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("📈 Key Levels Dashboard")
st.sidebar.markdown("---")

# Ticker input
ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA", max_chars=10).upper()

# Timeframe selection
timeframe_options = {
    '1D (Daily)': '1d',
    '4H (4 Hour)': '4h', 
    '1H (1 Hour)': '1h',
    '15m (15 Min)': '15m',
    '5m (5 Min)': '5m'
}
selected_tf_label = st.sidebar.selectbox("Chart Timeframe", list(timeframe_options.keys()))
selected_timeframe = timeframe_options[selected_tf_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")

# Toggle options
show_key_levels = st.sidebar.checkbox("Show Key Levels", value=True)
show_fibonacci = st.sidebar.checkbox("Show Fibonacci Levels", value=True)
show_support = st.sidebar.checkbox("Support Levels", value=True)
show_resistance = st.sidebar.checkbox("Resistance Levels", value=True)

# Importance filter
min_importance = st.sidebar.slider(
    "Minimum Importance", 
    min_value=1, 
    max_value=5, 
    value=1,
    help="1=5m, 2=15m, 3=1H, 4=4H, 5=1D"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Thresholds")

# Price threshold for merging
price_threshold = st.sidebar.number_input(
    "Price Merge Threshold ($)", 
    min_value=0.1, 
    max_value=5.0, 
    value=PRICE_THRESHOLD,
    step=0.1
)

# Fibonacci threshold
fib_threshold = st.sidebar.slider(
    "Fibonacci Threshold (%)",
    min_value=5,
    max_value=50,
    value=int(FIBONACCI_THRESHOLD * 100),
    step=5
) / 100

# Run analysis button
analyze_button = st.sidebar.button("🔄 Refresh Analysis", type="primary")

# Main content
st.title(f"📊 {ticker} Key Levels Analysis")

# Cache the analysis to avoid re-running on every interaction
@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_analysis(ticker, timeframes):
    """Run key levels analysis and return results."""
    kl = KeyLevels(ticker=ticker, use_alpaca=True)
    kl.find_all_key_levels(timeframes=timeframes)
    merged_df = kl.get_merged_levels(price_threshold=price_threshold)
    fib_df = kl.calculate_fibonacci_levels(
        merged_df=merged_df,
        fib_threshold=fib_threshold,
        min_importance=FIBONACCI_IMPORTANCE
    )
    return kl.timeframe_data, kl.all_levels_df, merged_df, fib_df

# Run or load analysis
with st.spinner(f"Analyzing {ticker}..."):
    try:
        timeframe_data, all_levels_df, merged_df, fib_df = run_analysis(
            ticker, 
            ['1d', '4h', '1h', '15m', '5m']
        )
        analysis_success = True
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
        analysis_success = False

if analysis_success and selected_timeframe in timeframe_data:
    # Get chart data for selected timeframe
    chart_df = timeframe_data[selected_timeframe]
    
    # Prepare x-axis data
    if 'Date' in chart_df.columns:
        chart_df['date_plot'] = pd.to_datetime(chart_df['Date'])
    elif 'Datetime' in chart_df.columns:
        chart_df['date_plot'] = pd.to_datetime(chart_df['Datetime'])
    else:
        chart_df['date_plot'] = pd.to_datetime(chart_df.index)
        
    # Date Range Slider
    min_date = chart_df['date_plot'].min().to_pydatetime()
    max_date = chart_df['date_plot'].max().to_pydatetime()
    
    col_range, _, _ = st.columns([1, 0.2, 0.2])
    with col_range:
        date_range = st.slider(
            "Zoom to Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
    
    # Filter data by date range
    mask = (chart_df['date_plot'] >= pd.Timestamp(date_range[0])) & (chart_df['date_plot'] <= pd.Timestamp(date_range[1]))
    filtered_chart_df = chart_df.loc[mask]
    
    if filtered_chart_df.empty:
        st.warning("No data in selected date range.")
    else:
        current_price = filtered_chart_df['close'].iloc[-1]
        
        # PROXIMITY FILTER: Filter levels within 25% of current price
        proximity_pct = 0.25
        min_price = current_price * (1 - proximity_pct)
        max_price = current_price * (1 + proximity_pct)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=filtered_chart_df['date_plot'],
            open=filtered_chart_df['open'],
            high=filtered_chart_df['high'],
            low=filtered_chart_df['low'],
            close=filtered_chart_df['close'],
            name=f"{ticker}",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ))
        
        # Get x-axis range
        x_start = filtered_chart_df['date_plot'].iloc[0]
        x_end = filtered_chart_df['date_plot'].iloc[-1]
        
        # Add key levels
        if show_key_levels and merged_df is not None and not merged_df.empty:
            # Filter by importance AND proximity
            filtered_levels = merged_df[
                (merged_df['importance'] >= min_importance) & 
                (merged_df['level_price'] >= min_price) & 
                (merged_df['level_price'] <= max_price)
            ]
            
            for _, row in filtered_levels.iterrows():
                level_type = row['type']
                
                if (level_type == 'support' and not show_support) or \
                   (level_type == 'resistance' and not show_resistance):
                    continue
                
                color = 'rgba(38, 166, 154, 0.9)' if level_type == 'support' else 'rgba(239, 83, 80, 0.9)'
                
                # Use solid line for high importance, dash for lower
                dash_style = 'solid' if row['importance'] >= 4 else 'dash'
                width = 2 if row['importance'] >= 4 else 1
                
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[row['level_price'], row['level_price']],
                    mode='lines+text',  # Add text label directly on line
                    line=dict(color=color, width=width, dash=dash_style),
                    text=[f"", f"${row['level_price']:.2f}"], # Label on right end
                    textposition="middle right",
                    textfont=dict(color=color, size=10),
                    name=f"${row['level_price']:.2f}",
                    showlegend=False,
                    hovertemplate=f"<b>${row['level_price']:.2f}</b><br>{row['type'].title()}<br>Touches: {row['touch_count']}<br>Importance: {row['importance']}<extra></extra>"
                ))
        
        # Add Fibonacci levels
    # Add Fibonacci levels
    if show_fibonacci and fib_df is not None and not fib_df.empty:
        # Get unique patterns and let user choose which to see
        if 'pattern_rank' in fib_df.columns:
            unique_patterns = fib_df.sort_values('pattern_rank').drop_duplicates(subset=['pattern_id'])
            
            # Sidebar: Pattern Selection
            st.sidebar.markdown("---")
            st.sidebar.subheader("📐 Fibonacci Patterns")
            
            # Multi-select for patterns (default to top 3)
            available_ranks = unique_patterns['pattern_rank'].tolist()
            default_ranks = available_ranks[:3] # Show top 3 by default
            
            selected_ranks = st.sidebar.multiselect(
                "Active Patterns (Ranked)",
                options=available_ranks,
                default=default_ranks,
                format_func=lambda x: f"Rank #{x} ({unique_patterns[unique_patterns['pattern_rank']==x].iloc[0]['pattern_id']})"
            )
            
            # Filter df by selected ranks
            active_fibs = fib_df[fib_df['pattern_rank'].isin(selected_ranks)].copy()
        else:
            active_fibs = fib_df.copy()
            selected_ranks = []

        # Proximity Filter for Fibs
        filtered_fibs = active_fibs[
            (active_fibs['fib_price'] >= min_price) & 
            (active_fibs['fib_price'] <= max_price)
        ]
        
        fib_colors = {
            0.382: 'rgba(255, 215, 0, 0.7)',  # Gold
            0.50: 'rgba(255, 165, 0, 0.7)',   # Orange
            0.618: 'rgba(255, 99, 71, 0.7)'   # Tomato
        }
        
        # If we have ranks, loop through selected ranks
        if selected_ranks:
            for rank in selected_ranks:
                pattern_fibs = filtered_fibs[filtered_fibs['pattern_rank'] == rank]
                
                for _, fib_row in pattern_fibs.iterrows():
                    color = fib_colors.get(fib_row['fib_ratio'], 'rgba(255, 215, 0, 0.6)')
                    
                    # Make Top 1 Rank bolder
                    width = 2 if rank == 1 else 1
                    dash_style = 'dot' if rank == 1 else 'dashdot'
                    
                    fig.add_trace(go.Scatter(
                        x=[x_start, x_end],
                        y=[fib_row['fib_price'], fib_row['fib_price']],
                        mode='lines',
                        line=dict(color=color, width=width, dash=dash_style), 
                        name=f"R#{rank} Fib {fib_row['fib_level']}",
                        showlegend=False,
                        hovertemplate=f"<b>Rank #{rank} | {fib_row['fib_level']}</b><br>${fib_row['fib_price']:.2f}<br>Pattern: {fib_row['pattern_id']}<extra></extra>"
                    ))
        else:
            # Fallback for no ranking column
            for _, fib_row in filtered_fibs.iterrows():
                color = fib_colors.get(fib_row['fib_ratio'], 'rgba(255, 215, 0, 0.6)')
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[fib_row['fib_price'], fib_row['fib_price']],
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    name=f"Fib {fib_row['fib_level']}",
                    showlegend=False
                ))
    
    # Update layout for Professional Appearance
    fig.update_layout(
        title=dict(
            text=f"{ticker} • {selected_tf_label} • ${current_price:.2f}",
            font=dict(size=20, color='#d1d4dc', family='Arial'),
            x=0.01,
            xanchor='left'
        ),
        xaxis_title="",
        yaxis_title="",
        template="plotly_dark",
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(family='Arial', size=11, color='#b2b5be'),
        xaxis=dict(
            gridcolor='rgba(42, 46, 57, 0.6)', # Subtle grid
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=False),
            showticklabels=True,
            showspikes=True, # Crosshair line
            spikemode='across',
            spikesnap='cursor',
            showline=False,
            tickformat='%Y-%m-%d\n%H:%M' if selected_timeframe in ['15m', '5m', '1h'] else '%Y-%m-%d',
        ),
        yaxis=dict(
            side='right', # Price scale on right
            gridcolor='rgba(42, 46, 57, 0.6)',
            showgrid=True,
            zeroline=False,
            tickprefix='$',
            tickformat='.2f',
            showspikes=True, # Crosshair line
            spikemode='across',
            spikesnap='cursor',
            range=[
                filtered_chart_df['low'].min() * 0.98, # Dynamic y-axis scaling
                filtered_chart_df['high'].max() * 1.02
            ]
        ),
        legend=dict(visible=False), # Hide legend completely for cleaner look
        height=700,
        margin=dict(l=10, r=60, t=50, b=20),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e222d',
            font_size=12,
            font_family='Arial',
            bordercolor='#2a2e39'
        )
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data tables (filtered)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🎯 Nearby Key Levels")
        if merged_df is not None and not merged_df.empty:
            # Show only levels in proximity
            display_df = merged_df[
                (merged_df['importance'] >= min_importance) &
                (merged_df['level_price'] >= min_price) & 
                (merged_df['level_price'] <= max_price)
            ].copy()
            display_df = display_df.sort_values('level_price', ascending=False)
            
            display_df['Price'] = display_df['level_price'].apply(lambda x: f"${x:.2f}")
            display_df['Type'] = display_df['type'].str.upper()
            display_df['Touches'] = display_df['touch_count']
            display_df['Imp'] = display_df['importance']
            
            st.dataframe(
                display_df[['Price', 'Type', 'Touches', 'Imp']].style.applymap(
                    lambda x: 'color: #ef5350' if x == 'RESISTANCE' else ('color: #26a69a' if x == 'SUPPORT' else ''),
                    subset=['Type']
                ),
                use_container_width=True,
                height=250,
                hide_index=True
            )
    
    with col2:
        st.markdown("##### 📐 Nearby Fibonacci Patterns (Ranked)")
        if fib_df is not None and not fib_df.empty and 'pattern_rank' in fib_df.columns:
            display_fib = fib_df.copy()
            # Filter proximity
            display_fib = display_fib[
                (display_fib['fib_price'] >= min_price) & 
                (display_fib['fib_price'] <= max_price)
            ].sort_values(['pattern_rank', 'fib_price'], ascending=[True, False])
            
            display_fib['Rank'] = display_fib['pattern_rank']
            display_fib['Price'] = display_fib['fib_price'].apply(lambda x: f"${x:.2f}")
            display_fib['Level'] = display_fib['fib_level']
            display_fib['Pattern'] = display_fib['pattern_id']
            
            st.dataframe(
                display_fib[['Rank', 'Price', 'Level', 'Pattern']],
                use_container_width=True,
                height=250,
                hide_index=True
            )

elif not analysis_success:
    st.warning("Unable to load data. Please check the ticker symbol and try again.")
else:
    st.warning(f"No data available for timeframe: {selected_tf_label}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly")
st.sidebar.caption("Data: Alpaca API / Yahoo Finance")
