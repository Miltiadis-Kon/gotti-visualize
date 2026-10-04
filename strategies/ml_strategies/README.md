# Machine Learning Strategy Suite (Roadmap & Status)

This directory houses planned predictive ML trading models designed to integrate with the Lumibot and `StrategyBaseplate` architecture.

## Architecture & Submodules

* **`linear_regression/`**: Planned ridge/lasso regression for dynamic mean-reversion forecasting and volatility boundary modeling.
* **`lstm/`**: Planned deep-learning sequence model (Long Short-Term Memory) for directional probability forecasting and multi-timeframe order flow state inference.

## Status

These directories are currently reserved scaffolding. Concrete implementations will subclass `StrategyBaseplate` and implement:
1. `filter()`: ML probability cutoff gate (e.g. `P(win) > 0.65`).
2. `setup()`: Feature engineering from normalized OHLCV, session VWAP, and ATR.
3. `get_position_sizing()`: Kelly-criterion or volatility-scaled allocation.
