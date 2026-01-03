"""
Trade Dashboard - Interactive TradingView-like Trade Visualization

Features:
- Load trades from backtest JSON output
- Resolution switching (1D, 4H, 1H, 15m, 5m)
- Trade markers with entry/exit points
- TP/SL lines visualization
- Trade zone shading (green for wins, red for losses)
- Key levels overlay (optional)
- Trade report table with full details
- Summary statistics

Run with: streamlit run plots/trade_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.trade_tracker import Trade, TradeTracker
from strategies.key_levels import (
    KeyLevels, 
    TIMEFRAME_COLORS, 
    TIMEFRAME_LOOKBACK,
    TIMEFRAME_IMPORTANCE,
    PRICE_THRESHOLD
)
from plots.chart_utils import (
    create_trade_chart,
    add_trade_markers,
    add_key_levels,
    COLORS
)
from plots.trade_report import (
    generate_trade_report,
    calculate_statistics,
    format_trade_for_display
)

# Page config
st.set_page_config(
    page_title="Trade Dashboard",
    page_icon="📊",
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
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 12px;
        color: #b2b5be;
    }
    .win { color: #26a69a !important; }
    .loss { color: #ef5350 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("📊 Trade Dashboard")
st.sidebar.markdown("---")

# Trade data source
st.sidebar.subheader("📁 Trade Data")

# Option to load trades
data_source = st.sidebar.radio(
    "Data Source",
    ["Load from File", "Run New Backtest"],
    index=0
)

trades = []
ticker = "NVDA"
tracker = None

if data_source == "Load from File":
    # Look for trade files in the project
    trades_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(trades_dir, "logs")
    
    def extract_ticker_from_filename(filename):
        """Extract ticker from filename patterns."""
        basename = os.path.basename(filename).replace('.json', '').replace('.csv', '')
        
        # Old format: TICKER_trades  
        if '_trades' in basename:
            return basename.split('_trades')[0]
        
        # New format: DDMM_HHMM_TICKER_STRATEGY
        parts = basename.split('_')
        if len(parts) >= 4:
            return parts[2]  # Third part is ticker
        
        return None
    
    # Find all trade files and extract tickers
    all_trade_files = []
    available_tickers = set()
    
    for search_dir in [trades_dir, logs_dir]:
        if os.path.exists(search_dir):
            for f in os.listdir(search_dir):
                # Skip Lumibot's native format
                if f.startswith('KeyLevels') or f.startswith('Simple5m') or f.startswith('Multi'):
                    continue
                if f.endswith('.json'):
                    filepath = os.path.join(search_dir, f)
                    # Old format: TICKER_trades.json
                    if '_trades.json' in f:
                        all_trade_files.append(filepath)
                        ticker_name = extract_ticker_from_filename(f)
                        if ticker_name:
                            available_tickers.add(ticker_name)
                    # New format: DDMM_HHMM_TICKER_STRATEGY.json
                    elif f.count('_') >= 3:
                        all_trade_files.append(filepath)
                        ticker_name = extract_ticker_from_filename(f)
                        if ticker_name:
                            available_tickers.add(ticker_name)
    
    # Add "All" option to tickers
    ticker_options = ["All"] + sorted(list(available_tickers))
    
    # Ticker filter dropdown
    if ticker_options:
        selected_ticker = st.sidebar.selectbox(
            "🎯 Filter by Ticker",
            ticker_options,
            index=0
        )
    else:
        selected_ticker = "All"
    
    # Filter files by selected ticker
    if selected_ticker == "All":
        trade_files = all_trade_files
    else:
        trade_files = [
            f for f in all_trade_files 
            if extract_ticker_from_filename(f) == selected_ticker
        ]
    
    # Sort by modification time (newest first)
    trade_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if trade_files:
        selected_file = st.sidebar.selectbox(
            "Select Trade File",
            trade_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        if selected_file and os.path.exists(selected_file):
            try:
                tracker = TradeTracker.load_from_json(selected_file)
                trades = tracker.trades
                ticker = tracker.ticker
                
                if trades:
                    st.sidebar.success(f"✅ Loaded {len(trades)} trades for {ticker}")
                else:
                    st.sidebar.warning("No trades in file")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                import traceback
                st.sidebar.code(traceback.format_exc())
    else:
        if selected_ticker != "All":
            st.sidebar.warning(f"No trade files for {selected_ticker}. Run a backtest first.")
        else:
            st.sidebar.warning("No trade files found. Run a backtest first.")
        
    # Manual file upload
    uploaded_file = st.sidebar.file_uploader("Or upload trades file", type=['json', 'csv'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                ticker = data.get('ticker', 'NVDA')
                trades = [Trade.from_dict(t) for t in data.get('trades', [])]
            else:
                # CSV upload - try Lumibot format
                df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)  # Reset for re-read
                trades, ticker = parse_lumibot_trades_csv(uploaded_file)
            
            if trades:
                st.sidebar.success(f"Loaded {len(trades)} trades from upload")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

else:
    # Run new backtest option
    ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        st.sidebar.info("Backtest feature coming soon. Run from command line for now.")
        st.sidebar.code("python strategies/key_levels_strategy.py")

st.sidebar.markdown("---")

# Timeframe selection
st.sidebar.subheader("📈 Chart Settings")
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
st.sidebar.subheader("👁️ Display Options")

# Toggle options
show_trades = st.sidebar.checkbox("Show Trade Markers", value=True)
show_tp_sl = st.sidebar.checkbox("Show TP/SL Lines", value=True)
show_key_levels = st.sidebar.checkbox("Show Key Levels", value=False)
show_support = st.sidebar.checkbox("Support Levels", value=True)
show_resistance = st.sidebar.checkbox("Resistance Levels", value=True)

if show_key_levels:
    min_importance = st.sidebar.slider(
        "Minimum Level Importance", 
        min_value=1, 
        max_value=5, 
        value=2,
        help="1=5m, 2=15m, 3=1H, 4=4H, 5=1D"
    )
else:
    min_importance = 1

# Main content
st.title(f"📊 {ticker} Trade Analysis")

# Statistics row
if trades:
    stats = calculate_statistics(trades)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Trades", stats['total_trades'])
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col3:
        pnl_color = "normal" if stats['total_pnl'] >= 0 else "inverse"
        st.metric("Total PnL", f"${stats['total_pnl']:,.2f}", delta_color=pnl_color)
    with col4:
        st.metric("Avg Win", f"${stats['avg_win']:,.2f}")
    with col5:
        st.metric("Avg Loss", f"${stats['avg_loss']:,.2f}")
    with col6:
        pf = stats['profit_factor']
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric("Profit Factor", pf_str)

# Fetch chart data
@st.cache_data(ttl=300)
def get_chart_data_for_range(ticker, timeframe, start_date, end_date):
    """Fetch OHLCV data for a specific date range."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        
        df = stock.history(
            start=start_date,
            end=end_date + timedelta(days=1),
            interval=timeframe,
            prepost=True
        )
        
        df.columns = [col.lower() for col in df.columns]
        df = df[df['volume'] != 0]
        df.reset_index(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=300)
def get_key_levels_data(ticker, timeframes):
    """Fetch key levels for the chart."""
    try:
        kl = KeyLevels(ticker=ticker, use_alpaca=False)
        kl.find_all_key_levels(timeframes=timeframes)
        merged_df = kl.get_merged_levels(price_threshold=PRICE_THRESHOLD)
        return merged_df
    except Exception as e:
        st.error(f"Error fetching levels: {e}")
        return None

# Determine date range from trades
if trades:
    trade_dates = [t.date_executed for t in trades if t.date_executed]
    if trade_dates:
        chart_start = min(trade_dates) - timedelta(days=2)
        chart_end = max(trade_dates) + timedelta(days=2)
    else:
        chart_start = datetime.now() - timedelta(days=30)
        chart_end = datetime.now()
else:
    chart_start = datetime.now() - timedelta(days=30)
    chart_end = datetime.now()

# Get chart data for trade date range
with st.spinner(f"Loading {ticker} data..."):
    chart_df = get_chart_data_for_range(ticker, selected_timeframe, chart_start, chart_end)
    
    if show_key_levels:
        levels_df = get_key_levels_data(ticker, ['1d', '4h', '1h', '15m', '5m'])
    else:
        levels_df = None

if chart_df is not None and not chart_df.empty:
    # Prepare date column - check both upper and lowercase (Yahoo uses lowercase after reset_index)
    date_col_found = None
    for col in ['Date', 'Datetime', 'date', 'datetime', 'index', 'Index']:
        if col in chart_df.columns:
            date_col_found = col
            break
    
    if date_col_found:
        chart_df['date_plot'] = pd.to_datetime(chart_df[date_col_found])
    else:
        chart_df['date_plot'] = pd.to_datetime(chart_df.index)
    
    # Date Range Slider
    min_date = chart_df['date_plot'].min().to_pydatetime()
    max_date = chart_df['date_plot'].max().to_pydatetime()
    
    # If we have trades, default to trade date range
    if trades:
        trade_dates = [t.date_executed for t in trades if t.date_executed]
        if trade_dates:
            trade_start = min(trade_dates) - timedelta(days=2)
            trade_end = max(trade_dates) + timedelta(days=2)
            default_range = (
                max(min_date, trade_start),
                min(max_date, trade_end)
            )
        else:
            default_range = (min_date, max_date)
    else:
        default_range = (min_date, max_date)
    
    date_range = st.slider(
        "📅 Date Range",
        min_value=min_date,
        max_value=max_date,
        value=default_range,
        format="YYYY-MM-DD"
    )
    
    # Filter data by date range
    mask = (chart_df['date_plot'] >= pd.Timestamp(date_range[0])) & \
           (chart_df['date_plot'] <= pd.Timestamp(date_range[1]))
    filtered_chart_df = chart_df.loc[mask].copy()
    
    if filtered_chart_df.empty:
        st.warning("No data in selected date range.")
    else:
        current_price = filtered_chart_df['close'].iloc[-1]
        
        # Create the chart
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=filtered_chart_df['date_plot'],
            open=filtered_chart_df['open'],
            high=filtered_chart_df['high'],
            low=filtered_chart_df['low'],
            close=filtered_chart_df['close'],
            name=ticker,
            increasing_line_color=COLORS['candle_up'],
            decreasing_line_color=COLORS['candle_down'],
            increasing_fillcolor=COLORS['candle_up'],
            decreasing_fillcolor=COLORS['candle_down']
        ))
        
        x_start = filtered_chart_df['date_plot'].iloc[0]
        x_end = filtered_chart_df['date_plot'].iloc[-1]
        
        # Add key levels
        if show_key_levels and levels_df is not None and not levels_df.empty:
            # Proximity filter
            proximity_pct = 0.25
            min_price = current_price * (1 - proximity_pct)
            max_price = current_price * (1 + proximity_pct)
            
            filtered_levels = levels_df[
                (levels_df['importance'] >= min_importance) &
                (levels_df['level_price'] >= min_price) &
                (levels_df['level_price'] <= max_price)
            ]
            
            for _, row in filtered_levels.iterrows():
                level_type = row['type']
                
                if (level_type == 'support' and not show_support) or \
                   (level_type == 'resistance' and not show_resistance):
                    continue
                
                color = COLORS['support'] if level_type == 'support' else COLORS['resistance']
                dash_style = 'solid' if row['importance'] >= 4 else 'dash'
                width = 2 if row['importance'] >= 4 else 1
                
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[row['level_price'], row['level_price']],
                    mode='lines+text',
                    line=dict(color=color, width=width, dash=dash_style),
                    text=["", f"${row['level_price']:.2f}"],
                    textposition="middle right",
                    textfont=dict(color=color, size=10),
                    showlegend=False,
                    hovertemplate=f"<b>${row['level_price']:.2f}</b><br>{row['type'].title()}<br>Importance: {row['importance']}<extra></extra>"
                ))
        
        # Add trades
        if show_trades and trades:
            # Filter trades in date range
            visible_trades = [
                t for t in trades 
                if t.date_executed and date_range[0] <= t.date_executed <= date_range[1]
            ]
            
            for trade in visible_trades:
                entry_date = trade.date_executed
                exit_date = trade.date_completed if trade.date_completed else x_end
                
                # Trade zone shading
                if trade.date_completed:
                    zone_color = COLORS['trade_win'] if trade.is_winner else COLORS['trade_loss']
                    fig.add_vrect(
                        x0=entry_date,
                        x1=exit_date,
                        fillcolor=zone_color,
                        layer="below",
                        line_width=0,
                    )
                
                # TP/SL lines
                if show_tp_sl:
                    # Take Profit line
                    fig.add_trace(go.Scatter(
                        x=[entry_date, exit_date],
                        y=[trade.take_profit, trade.take_profit],
                        mode='lines',
                        line=dict(color=COLORS['tp_line'], width=1, dash='dash'),
                        showlegend=False,
                        hovertemplate=f"<b>Take Profit</b><br>${trade.take_profit:.2f}<extra></extra>"
                    ))
                    
                    # Stop Loss line
                    fig.add_trace(go.Scatter(
                        x=[entry_date, exit_date],
                        y=[trade.stop_loss, trade.stop_loss],
                        mode='lines',
                        line=dict(color=COLORS['sl_line'], width=1, dash='dash'),
                        showlegend=False,
                        hovertemplate=f"<b>Stop Loss</b><br>${trade.stop_loss:.2f}<extra></extra>"
                    ))
                
                # Entry marker
                fig.add_trace(go.Scatter(
                    x=[entry_date],
                    y=[trade.entry_price],
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-up',
                        size=14,
                        color=COLORS['entry_marker'],
                        line=dict(width=1, color='white')
                    ),
                    text=[f"BUY ${trade.entry_price:.2f}"],
                    textposition="top center",
                    textfont=dict(color=COLORS['entry_marker'], size=10),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>ENTRY</b><br>"
                        f"Price: ${trade.entry_price:.2f}<br>"
                        f"Qty: {trade.quantity}<br>"
                        f"TP: ${trade.take_profit:.2f}<br>"
                        f"SL: ${trade.stop_loss:.2f}"
                        f"<extra></extra>"
                    )
                ))
                
                # Exit marker
                if trade.date_completed and trade.exit_price:
                    exit_color = COLORS['entry_marker'] if trade.is_winner else COLORS['exit_marker']
                    pnl_text = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
                    
                    fig.add_trace(go.Scatter(
                        x=[trade.date_completed],
                        y=[trade.exit_price],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down',
                            size=14,
                            color=exit_color,
                            line=dict(width=1, color='white')
                        ),
                        text=[f"{trade.exit_reason} {pnl_text}"],
                        textposition="bottom center",
                        textfont=dict(color=exit_color, size=10),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>EXIT ({trade.exit_reason})</b><br>"
                            f"Price: ${trade.exit_price:.2f}<br>"
                            f"PnL: {pnl_text}"
                            f"<extra></extra>"
                        )
                    ))
        
        # Layout
        tick_format = '%Y-%m-%d\n%H:%M' if selected_timeframe in ['15m', '5m', '1h'] else '%Y-%m-%d'
        
        fig.update_layout(
            title=dict(
                text=f"{ticker} • {selected_tf_label} • ${current_price:.2f}",
                font=dict(size=20, color=COLORS['text'], family='Arial'),
                x=0.01,
                xanchor='left'
            ),
            xaxis_title="",
            yaxis_title="",
            template="plotly_dark",
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['paper'],
            font=dict(family='Arial', size=11, color=COLORS['text_secondary']),
            xaxis=dict(
                gridcolor=COLORS['grid'],
                showgrid=True,
                zeroline=False,
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                tickformat=tick_format,
            ),
            yaxis=dict(
                side='right',
                gridcolor=COLORS['grid'],
                showgrid=True,
                zeroline=False,
                tickprefix='$',
                tickformat='.2f',
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                range=[
                    filtered_chart_df['low'].min() * 0.98,
                    filtered_chart_df['high'].max() * 1.02
                ]
            ),
            legend=dict(visible=False),
            height=650,
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
        
        # Trade report table
        if trades:
            st.markdown("---")
            st.subheader("📋 Trade Report")
            
            # Generate report DataFrame
            report_df = generate_trade_report(trades)
            
            # Format for display
            display_df = report_df.copy()
            display_df['date_executed'] = pd.to_datetime(display_df['date_executed']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['date_completed'] = display_df['date_completed'].apply(
                lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'Open'
            )
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else '-')
            display_df['take_profit'] = display_df['take_profit'].apply(lambda x: f"${x:.2f}")
            display_df['stop_loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}")
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+,.2f}" if pd.notnull(x) else '-')
            display_df['total_pnl'] = display_df['total_pnl'].apply(lambda x: f"${x:+,.2f}" if pd.notnull(x) else '-')
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'date_executed': 'Date Executed',
                'ticker': 'Ticker',
                'type': 'Type',
                'date_completed': 'Date Completed',
                'entry_price': 'Entry',
                'exit_price': 'Exit',
                'take_profit': 'TP',
                'stop_loss': 'SL',
                'exit_reason': 'Reason',
                'pnl': 'PnL',
                'total_pnl': 'Cumulative PnL'
            })
            
            # Select columns
            cols = ['Date Executed', 'Ticker', 'Type', 'Entry', 'Exit', 
                    'TP', 'SL', 'Reason', 'Date Completed', 'PnL', 'Cumulative PnL']
            
            st.dataframe(
                display_df[cols],
                use_container_width=True,
                height=300,
                hide_index=True
            )
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = report_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name=f"{ticker}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col2:
                if tracker:
                    json_data = json.dumps({
                        'ticker': tracker.ticker,
                        'total_pnl': tracker.total_pnl,
                        'statistics': tracker.get_statistics(),
                        'trades': [t.to_dict() for t in tracker.trades]
                    }, indent=2, default=str)
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        file_name=f"{ticker}_trades_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

else:
    st.warning("Unable to load chart data. Please check the ticker symbol.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly")
st.sidebar.caption("Data: Yahoo Finance")
