"""
Debug Runner for SmartMoneyConceptsStrategy
============================================
Runs 1 month of NVDA to see all log messages from SmartMoneyConceptsStrategy.
"""

import sys
import os
from datetime import datetime
import pandas as pd
import yfinance as yf

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None
import lumibot.data_sources.data_source_backtesting
lumibot.data_sources.data_source_backtesting.print_progress_bar = lambda *args, **kwargs: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumibot.entities import Asset, Data
from lumibot.backtesting import PandasDataBacktesting
from strategies.smc.smc_strategy import SmartMoneyConceptsStrategy

df = yf.download("NVDA", interval="5m", period="60d", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0].lower() for c in df.columns]
else:
    df.columns = [c.lower() for c in df.columns]

asset = Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)
quote = Asset(symbol="USD", asset_type="forex")
pandas_data = {asset: Data(asset, df, timestep="minute", quote=quote)}

start_dt = df.index[100]
end_dt = df.index[-1]

print(f"Starting debug backtest from {start_dt} to {end_dt} ({len(df)} candles)...")

results, strategy = SmartMoneyConceptsStrategy.run_backtest(
    datasource_class=PandasDataBacktesting,
    backtesting_start=start_dt,
    backtesting_end=end_dt,
    pandas_data=pandas_data,
    budget=100_000.0,
    parameters={"Ticker": asset, "Plot": False},
    show_plot=False
)

print(f"\n--- Strategy Execution Summary ---")
print(f"Total Trade Records: {len(strategy.trade_records)}")
for tr in strategy.trade_records:
    print(tr)

if strategy.active_trade:
    print(f"Active trade still open at end of backtest: {strategy.active_trade}")
