# Graph Report - gotti-visualize  (2026-10-06)

## Corpus Check
- 73 files · ~2,489,829 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .csv 5, (none) 4, .example 1)

## Summary
- 1121 nodes · 2067 edges · 80 communities (30 shown, 50 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 136 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c2060946`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- BaseChartRenderer
- api.py
- BaseKeyLevelsStrategy
- ChartBuilder
- levels_visualizer.py
- MultiTimeframeTrendContinuationStrategy
- MultiTimeframeKeyLevelsStrategy
- SignalStore
- Signal
- typing
- SwingStrategyBase
- TradeTracker
- StrategyBaseplate
- KeyLevels
- SignalLoader
- trade_dashboard.py
- DataFetcher
- BACKTESTING_ARCHITECTURE.md - LumiBot Backtesting Architecture
- KeyLevelDetector
- PBInvestingVWAPEMA
- FibonacciDetector
- Trade
- create_params_db.py
- key_levels/__init__.py
- AnalysisResult
- os
- LiquiditySweepStrategy
- SMCIntradayBiasStrategy
- PlotSpec
- smc_tranche_manager.py
- app_backtest.py
- session_killzones.py
- AlpacaSMCBridge
- mysql_candle_feed.py
- smc/__init__.py
- smc_structure.py
- M5POISelector
- smc_liquidity.py
- SmartMoneyConceptsStrategy
- Docker Setup (Recommended — runs 24/7)
- Complete Smart Money Concepts (SMC) Trading Strategy Manual
- Signal Processor
- Key Levels Trading Strategy — In-Depth Documentation
- m5_poi.py
- smc_bias_model/__init__.py
- M1CHoCHDetector
- SwingPoint
- .close_trade
- MACDTradingStrategy
- 4. MultiTimeframeKeyLevelsStrategy — The Concrete Implementation
- pandas
- 3.7 Abstract Methods (Template Method Pattern)
- Querying Signals
- bias_order_manager.py

## God Nodes (most connected - your core abstractions)
1. `StrategyBaseplate` - 41 edges
2. `BaseKeyLevelsStrategy` - 38 edges
3. `PlotSpec` - 36 edges
4. `SwingStrategyBase` - 30 edges
5. `SignalStore` - 29 edges
6. `TradeTracker` - 28 edges
7. `BaseChartRenderer` - 23 edges
8. `ChartBuilder` - 21 edges
9. `CandlestickLayer` - 20 edges
10. `Signal` - 19 edges

## Surprising Connections (you probably didn't know these)
- `4.5 `get_exit_signal()` — TP/SL Monitoring` --references--> `TradeTracker`  [INFERRED]
  STRATEGY_DOCUMENTATION.md → strategies/trade_tracker.py
- `3.8 Helper Methods` --references--> `KeyLevels`  [INFERRED]
  STRATEGY_DOCUMENTATION.md → strategies/key_levels/__init__.py
- `1. Architecture Overview` --references--> `Trade`  [INFERRED]
  STRATEGY_DOCUMENTATION.md → strategies/trade_tracker.py
- `1. Architecture Overview` --references--> `TradeTracker`  [INFERRED]
  STRATEGY_DOCUMENTATION.md → strategies/trade_tracker.py
- `2. Lumibot Lifecycle Methods — How They Apply` --references--> `TradeTracker`  [INFERRED]
  STRATEGY_DOCUMENTATION.md → strategies/trade_tracker.py

## Import Cycles
- None detected.

## Communities (80 total, 50 thin omitted)

### Community 0 - "BaseChartRenderer"
Cohesion: 0.17
Nodes (7): CandlestickLayer, IndicatorLayer, KeyLevelsLayer, PlotLayer, TradeMarkersLayer, BaseChartRenderer, get_renderer()

### Community 1 - "api.py"
Cohesion: 0.08
Nodes (13): calculate_risk_levels(), chart_redirect(), get_db_connection(), get_db_data(), get_key_levels(), get_ohlcv(), get_risk_styles(), health_check() (+5 more)

### Community 4 - "levels_visualizer.py"
Cohesion: 0.21
Nodes (4): get_price_data_single_day(), load_levels_data(), get_chart_data_for_range(), get_key_levels_data()

### Community 6 - "MultiTimeframeKeyLevelsStrategy"
Cohesion: 0.06
Nodes (10): MultiTimeframeKeyLevelsStrategy, PlottableStrategyMixin, 1. Key Levels & Multi-Timeframe Fibonacci Framework, 2. Thomas Bulkowski Day Trading Strategies (5-Minute Intraday), 4. Baseplate & Shared Utilities, Architecture Overview, `BaseKeyLevelsStrategy` (`base_key_levels_strategy.py`), `MultiTimeframeKeyLevelsStrategy` (`multi_tf_strategy.py`) (+2 more)

### Community 10 - "typing"
Cohesion: 0.11
Nodes (4): ATRRiskLevels, calculate_atr(), calculate_trailing_stop(), optimize_sl_tp()

### Community 11 - "SwingStrategyBase"
Cohesion: 0.09
Nodes (10): 3. Thomas Bulkowski Swing Trading Strategies (Daily - 1D), FibRetraceSwing, GapSetupLongSwing, GapSetupShortSwing, HCRBreakoutLongSwing, HCRBreakoutShortSwing, RetraceLongSwing, RetraceShortSwing (+2 more)

### Community 12 - "TradeTracker"
Cohesion: 0.12
Nodes (9): TradeTracker, 3.1 Class Definition & Parameters, 3.2 `initialize()` — One-Time Setup, 3.3 `on_trading_iteration()` — The Main Loop, 3.5 `after_market_closes()` — End-of-Day Persistence, 3.6 `on_abrupt_closing()` — Graceful Shutdown, 3.8 Helper Methods, 3. BaseKeyLevelsStrategy — The Abstract Foundation (+1 more)

### Community 13 - "StrategyBaseplate"
Cohesion: 0.05
Nodes (5): LanceBreitsteinScalping, Architecture & Submodules, Machine Learning Strategy Suite (Roadmap & Status), Status, StrategyBaseplate

### Community 14 - "KeyLevels"
Cohesion: 0.14
Nodes (3): run_analysis(), KeyLevelAnalyzer, KeyLevels

### Community 16 - "trade_dashboard.py"
Cohesion: 0.14
Nodes (7): extract_ticker_from_filename(), calculate_statistics(), export_trades_csv(), format_trade_for_display(), generate_trade_report(), print_trade_summary(), print_trade_table()

### Community 18 - "BACKTESTING_ARCHITECTURE.md - LumiBot Backtesting Architecture"
Cohesion: 0.06
Nodes (31): 1. BacktestingBroker (`backtesting/backtesting_broker.py`), 2. Data Source Hierarchy, 3. Yahoo Finance (`yahoo_backtesting.py` → `yahoo_data.py` → `yahoo_helper.py`), 4. ThetaData (`thetadata_backtesting_pandas.py` → `thetadata_helper.py`), 5. Polygon (`polygon_backtesting.py` → `polygon_helper.py`), Backtest Results Don't Match Between Data Sources, BACKTESTING_ARCHITECTURE.md - LumiBot Backtesting Architecture, Cache Issues (+23 more)

### Community 22 - "Trade"
Cohesion: 0.11
Nodes (4): Trade, 5.1 Trade Dataclass, 5.2 TradeTracker Class, 5. TradeTracker — Trade Recording & Analytics

### Community 25 - "key_levels/__init__.py"
Cohesion: 0.15
Nodes (5): FibonacciPattern, find_fibonacci_levels(), find_key_levels(), KeyLevel, merge_key_levels()

### Community 30 - "SMCIntradayBiasStrategy"
Cohesion: 0.13
Nodes (4): BiasOrderManager, BiasTradeRecord, BiasTradeTranche, SMCIntradayBiasStrategy

### Community 40 - "smc_tranche_manager.py"
Cohesion: 0.12
Nodes (3): PositionTranche, SMCTradeLifecycle, TrancheStatus

### Community 41 - "app_backtest.py"
Cohesion: 0.13
Nodes (4): generate_plotly_chart(), log_cb(), run_one_backtest(), _safe_signal()

### Community 43 - "session_killzones.py"
Cohesion: 0.16
Nodes (3): LiquidationStatus, SessionExtremes, SessionKillzoneManager

### Community 46 - "smc/__init__.py"
Cohesion: 0.18
Nodes (5): detect_fvgs(), FairValueGap, filter_fvgs_by_equilibrium(), FVGType, update_fvg_mitigation()

### Community 47 - "smc_structure.py"
Cohesion: 0.20
Nodes (6): detect_structure_shifts(), MarketStructure, StructureShiftEvent, StructureType, SwingType, TrendBias

### Community 49 - "smc_liquidity.py"
Cohesion: 0.16
Nodes (5): detect_equal_highs_lows(), extract_pdh_pdl(), find_liquidity_pools(), LiquidityPool, LiquidityType

### Community 50 - "SmartMoneyConceptsStrategy"
Cohesion: 0.17
Nodes (3): evaluate_3r_veto(), SmartMoneyConceptsStrategy, SMCTrancheManager

### Community 51 - "Docker Setup (Recommended — runs 24/7)"
Cohesion: 0.13
Nodes (14): Common errors and workarounds, Contribution, Docker Setup (Recommended — runs 24/7), Fibonacci levels, IMPORTANT, License, Overview, Prerequisites (+6 more)

### Community 52 - "Complete Smart Money Concepts (SMC) Trading Strategy Manual"
Cohesion: 0.13
Nodes (14): 1. The Entry, 2. The Stop Loss (Invalidation Point), 3. Risk-to-Reward (The 3R Minimum & The Veto Rule), Complete Smart Money Concepts (SMC) Trading Strategy Manual, Core Philosophy, Fair Value Gap (FVG) Mastery, Identifying a Valid Change of Character (CHoCH), Part 1: The 8-Step Macro-to-Micro Workflow (+6 more)

### Community 53 - "Signal Processor"
Cohesion: 0.14
Nodes (11): Accessors, Adding Custom Filters (Strategy Integration), Architecture, Converting to Pandas, CSV Format Expected, Grouping, Overview, Quick Start (+3 more)

### Community 54 - "Key Levels Trading Strategy — In-Depth Documentation"
Cohesion: 0.15
Nodes (7): 1. Architecture Overview, 2. Lumibot Lifecycle Methods — How They Apply, 6. Execution Flow — End to End, 7. Key Level Caching Mechanism, 9. Position Sizing Formula, Key Levels Trading Strategy — In-Depth Documentation, Table of Contents

### Community 56 - "smc_bias_model/__init__.py"
Cohesion: 0.15
Nodes (5): BiasDirection, ExternalBoxRange, ExternalSwingPoint, M15BoxStructureDetector, SwingPointType

### Community 57 - "M1CHoCHDetector"
Cohesion: 0.22
Nodes (3): M1CHoCHConfirmation, M1CHoCHDetector, MicroSwingPoint

### Community 61 - "4. MultiTimeframeKeyLevelsStrategy — The Concrete Implementation"
Cohesion: 0.25
Nodes (8): 4.1 Strategy Parameters, 4.2 `get_entry_signal()` — LONG + SHORT Logic, 4.3 `_check_long_entry()` — Buying at Support, 4.4 `_check_short_entry()` — Selling at Resistance, 4.5 `get_exit_signal()` — TP/SL Monitoring, 4.6 TP/SL Threshold Calculations, 4.7 `run_backtest()` — Backtesting Runner, 4. MultiTimeframeKeyLevelsStrategy — The Concrete Implementation

### Community 68 - "3.7 Abstract Methods (Template Method Pattern)"
Cohesion: 0.67
Nodes (3): 3.7 Abstract Methods (Template Method Pattern), `get_entry_signal(current_price, support_levels, resistance_levels) → Optional[Dict]`, `get_exit_signal(current_price, position, entry_support, target_resistance) → Optional[str]`

### Community 78 - "Querying Signals"
Cohesion: 0.29
Nodes (7): By Date, By Position, By Ticker, Chaining Filters, Custom Filters, Querying Signals, Sentiment Filters

## Knowledge Gaps
- **72 isolated node(s):** `Overview`, `Directory Structure`, `Data Flow for Backtesting`, `1. BacktestingBroker (`backtesting/backtesting_broker.py`)`, `2. Data Source Hierarchy` (+67 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 567 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **50 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StrategyBaseplate` connect `StrategyBaseplate` to `BaseChartRenderer`, `BaseKeyLevelsStrategy`, `MultiTimeframeKeyLevelsStrategy`, `PlotSpec`, `typing`, `SwingStrategyBase`, `SmartMoneyConceptsStrategy`, `PBInvestingVWAPEMA`, `os`, `LiquiditySweepStrategy`, `SMCIntradayBiasStrategy`?**
  _High betweenness centrality (0.202) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `StrategyBaseplate` (e.g. with `Machine Learning Strategy Suite (Roadmap & Status)` and `Status`) actually correct?**
  _`StrategyBaseplate` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Overview`, `Directory Structure`, `Data Flow for Backtesting` to the rest of the system?**
  _72 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `api.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0773109243697479 - nodes in this community are weakly interconnected._
- **Why does `BaseKeyLevelsStrategy` connect `BaseKeyLevelsStrategy` to `BaseChartRenderer`, `.get_entry_signal`, `.initialize`, `MultiTimeframeKeyLevelsStrategy`, `PlotSpec`, `typing`, `TradeTracker`, `StrategyBaseplate`, `Key Levels Trading Strategy — In-Depth Documentation`, `.close_trade`, `MACDTradingStrategy`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `BaseKeyLevelsStrategy` (e.g. with `CandlestickLayer` and `KeyLevelsLayer`) actually correct?**
  _`BaseKeyLevelsStrategy` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `BaseKeyLevelsStrategy` be split into smaller, more focused modules?**
  _Cohesion score 0.08108108108108109 - nodes in this community are weakly interconnected._