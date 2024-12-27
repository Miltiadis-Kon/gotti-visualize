import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from strategies.sma_cross_multisymbol import ChillGuy, run_backtest, run_live,run_live_crypto
from multiprocessing import Process

# Load environment variables
load_dotenv()

# Configuration
ALPACA_CONFIG = {
    "API_KEY": os.getenv("APCA_API_KEY_PAPER"),
    "API_SECRET": os.getenv("APCA_API_SECRET_KEY_PAPER"),
    "PAPER": True,
}

# Streamlit app
st.set_page_config(page_title="Gotti Trading Bot", layout="wide")
st.title("Gotti Trading Dashboard")

# Sidebar
st.sidebar.header("Trading Configuration")
trading_mode = st.sidebar.radio("Select Trading Mode", ["Live Trading", "Backtesting"])

# Default tickers
default_tickers = ['NVDA', 'AAPL', 'AMZN', 'TSLA', 'MARA']
tickers = st.sidebar.multiselect("Select Tickers", default_tickers, default=default_tickers)

if trading_mode == "Live Trading":
    st.header("Live Trading")
    
    if st.button("Start Live Trading"):
        with st.spinner("Starting live trading..."):
            try:
                process = Process(target=run_live, args=(tickers,))
                process.start()
                st.success("Live trading started!")
            except Exception as e:
                st.error(f"Error starting live trading: {str(e)}")

else:
    st.header("Backtesting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime(2023, 10, 23)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime(2024, 10, 23)
        )
    
    budget = st.number_input("Initial Budget", value=10000, step=1000)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                results = run_backtest(
                    tickers=tickers,
                    backtesting_start=datetime.combine(start_date, datetime.min.time()),
                    backtesting_end=datetime.combine(end_date, datetime.min.time()),
                    budget=budget
                )
                
                # Display results
                st.subheader("Backtest Results")
                st.json(results)
                
                # Add visualization if your backtest returns chart data
                if hasattr(results, 'chart_data'):
                    st.line_chart(results.chart_data)
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Trading Bot Status")
st.sidebar.markdown("Connected to Alpaca Paper Trading")