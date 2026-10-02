# Strategy Architecture & Algorithmic Catalog

This directory houses the quantitative trading strategies, technical level detection engines, trade tracking infrastructure, and backtesting runners for the `gotti-visualize` trading platform.

---

## Architecture Overview

The system is organized into four complementary layers:

```
strategies/
├── base_key_levels_strategy.py   # Abstract Base Class (Template Method pattern, Lumibot integration)
├── multi_tf_strategy.py          # Concrete Multi-Timeframe S/R + Fibonacci Trading Strategy (v2)
├── strat_baseplate.py            # Baseplate/Template strategy with VIX volume filter & GTD orders
├── plot_mixin.py                 # PlottableStrategyMixin for standalone Plotly HTML chart generation
├── trade_tracker.py              # Trade dataclass and TradeTracker ledger & performance statistics
│
├── key_levels/                   # Algorithmic Detection & Orchestration Engine
│   ├── analyzer.py               # DataFetcher & multi-resolution orchestrator (analyze())
│   ├── key_levels.py             # Pivot-point S/R detection, clustering, touch counting
│   ├── fibonacci_levels.py       # Swing detection, Fibonacci retracements, trade setups
│   └── chart.py                  # Interactive Plotly candlestick visualizations with rainbow fibs
│
├── book_based_2/                 # Thomas Bulkowski 5-Minute Intraday Strategies (Exact Implementations)
│   ├── hcr_breakout_long.py      # Horizontal Consolidation Region Breakout (Long)
│   ├── hcr_breakout_short.py     # Horizontal Consolidation Region Breakout (Short)
│   ├── gap_setup_long.py         # Opening Gap-Down Fade / Fill (Long)
│   ├── gap_setup_short.py        # Opening Gap-Up Fade / Fill (Short)
│   ├── fib_retrace_swing.py      # 38.2%-61.8% Intraday Fibonacci Retracement
│   ├── retrace_long.py           # 3-Candle Downtrend Retracement (Long)
│   ├── retrace_short.py          # 3-Candle Uptrend Retracement (Short)
│   ├── rsi_setup.py              # RSI(14) Oversold Bounce (30 Crossover)
│   └── run_*.py                  # Multi-ticker intraday backtest runners & metric scripts
│
└── swing/                        # Thomas Bulkowski Daily (1D) Swing Trading Variations
    ├── hcr_breakout_long.py      # 15-Day HCR Breakout with ATR Trailing Stop
    ├── hcr_breakout_short.py     # 15-Day HCR Breakdown with ATR Trailing Stop
    ├── gap_setup_long.py         # Daily Gap-Down Fade with ATR Stop
    ├── gap_setup_short.py        # Daily Gap-Up Fade with ATR Stop
    ├── fib_retrace_swing.py      # Daily 10-Day Swing Golden Pocket Retracement
    ├── retrace_long.py           # 3 Consecutive Down Days Reversal
    ├── retrace_short.py          # 3 Consecutive Up Days Reversal
    ├── rsi_setup.py              # Daily RSI(14) Oversold Swing
    └── run_swing_backtests.py    # 1-Year Multi-Ticker Swing Runner (NVDA, MARA, PLTR)
```

---

## 1. Key Levels & Multi-Timeframe Fibonacci Framework

### `BaseKeyLevelsStrategy` (`base_key_levels_strategy.py`)
- **Pattern**: Template Method Pattern (GoF) extending `lumibot.strategies.Strategy` and `PlottableStrategyMixin`.
- **Infrastructure Owned**:
  - Automatic order management and bracket submission (Entry, Stop Loss, Take Profit).
  - Risk-based position sizing (`RISK_PERCENT`, default 2% or 10%).
  - In-memory date-based caching (`_cached_levels`) to avoid redundant recalculation across backtest bars.
  - Duplicate trade suppression (`_entered_levels`).
  - End-of-day trade ledger export (JSON + CSV in `logs/trades/`).
  - Graceful crash & abrupt shutdown handling (`on_abrupt_closing`).
- **Abstract Hooks for Subclasses**:
  - `get_entry_signal(current_price, support_levels, resistance_levels)`
  - `get_exit_signal(current_price, position)`

### `MultiTimeframeKeyLevelsStrategy` (`multi_tf_strategy.py`)
- **Core Logic**:
  - **Dynamic Multi-Resolution Analysis**: Every 15 minutes (3 iterations × 5-min sleeptime), invokes `key_levels.analyzer.analyze()` across `['1D', '4H', '15m']`.
  - **Primary Signal (Fibonacci Retracement)**:
    - **Uptrend**: Price reaches 0.618 retracement of the swing move -> `BUY`. Take Profit at swing high (0.00 retracement), Stop Loss at 0.786 retracement.
    - **Downtrend**: Price reaches 0.618 retracement -> `SELL / SHORT`. Take Profit at swing low, Stop Loss at 0.786 retracement.
  - **Secondary Signal (Support / Resistance)**:
    - Buys within `ENTRY_THRESHOLD` (0.5%) of confirmed Support (minimum importance = 3).
    - Shorts within `ENTRY_THRESHOLD` (0.5%) of confirmed Resistance.
    - Filtered by `MIN_RISK_REWARD >= 1.5`.
  - **Noise Reduction Engine**:
    - If both Fibonacci and S/R trigger simultaneously, Fibonacci takes absolute precedence.
    - If an S/R signal forms inside an active Fibonacci entry zone (`FIB_SR_PROXIMITY = 2%`), S/R is suppressed until the exact Fibonacci price is met.

### Analytical Engines (`strategies/key_levels/`)
- **`KeyLevelDetector` (`key_levels.py`)**: Uses multi-bar lookback pivot detection (`_detect_pivot`) to locate local maxima and minima. Groups nearby levels using percentage clustering (`merge_key_levels`), tracking touch counts and assigning resolution weights (1D=5, 4H=4, 1H=3, 15m=2, 5m=1).
- **`FibonacciDetector` (`fibonacci_levels.py`)**: 40-bar swing window detector that computes standard Fibonacci ratios (`0.0, 0.236, 0.382, 0.50, 0.618, 0.786, 1.0`). Derives structured trade setups (`get_fibonacci_trade_setups`) with explicit entry, stop loss, and target prices.
- **`DataFetcher` & `analyze()` (`analyzer.py`)**: Automated data fetcher (supporting Alpaca Market Data API and Yahoo Finance) supporting historical backtesting via `as_of_date`.
- **`chart.py`**: Visualizer that renders TradingView-style multi-color Fibonacci bands and blue-shaded S/R levels.

---

## 2. Thomas Bulkowski Day Trading Strategies (5-Minute Intraday)

Directory: `strategies/book_based_2/`  
Reference: *Swing and Day Trading: Evolution of a Trader* and ThePatternSite.com.  
Timeframe: 5-Minute candles (`sleeptime = "5M"`). Local 1-minute to 5-minute resampling. Strict $0.01 breakout and stop-loss offsets.

| Strategy Class | File | Signal Setup | Entry Trigger | Exit & Stop Loss |
|---|---|---|---|---|
| **`HCRBreakoutLong`** | `hcr_breakout_long.py` | Horizontal Consolidation Region (HCR) over 15 bars (75m), range $\le 0.3\%$ | Buy when price breaks $\$0.01$ above HCR high | Initial stop: $\$0.01$ below HCR low. Trailing stop: $\$0.01$ below prior 5m candle low |
| **`HCRBreakoutShort`** | `hcr_breakout_short.py` | Horizontal Consolidation Region (HCR) over 15 bars (75m), range $\le 0.3\%$ | Short when price breaks $\$0.01$ below HCR low | Initial stop: $\$0.01$ above HCR high. Trailing stop: $\$0.01$ above prior 5m candle high |
| **`OpeningGapLong`** | `gap_setup_long.py` | Opening gap down (Today Open < Yesterday Close) | Buy when price breaks $\$0.01$ above high of first 5m candle | Target: Yesterday close (gap fill). Stop: $\$0.01$ below first candle low |
| **`OpeningGapShort`** | `gap_setup_short.py` | Opening gap up (Today Open > Yesterday Close) | Short when price breaks $\$0.01$ below low of first 5m candle | Target: Yesterday close (gap fill). Stop: $\$0.01$ above first candle high |
| **`FibRetraceSwing`** | `fib_retrace_swing.py` | Straight uprun (3 consecutive higher closes) followed by 38.2%–61.8% pullback | Buy when price enters zone and current 5m candle closes green | Target: Prior swing high. Stop: $\$0.01$ below swing low |
| **`RetraceLong`** | `retrace_long.py` | Micro-downtrend (3 consecutive red 5m candles) | Buy when price crosses $\$0.01$ above 3rd candle's high | Stop: $\$0.01$ below swing low. Target: $+1.0\%$ profit target / trail |
| **`RetraceShort`** | `retrace_short.py` | Micro-uptrend (3 consecutive green 5m candles) | Short when price crosses $\$0.01$ below 3rd candle's low | Stop: $\$0.01$ above swing high. Target: $+1.0\%$ profit target / trail |
| **`RSISetup`** | `rsi_setup.py` | 14-period RSI drops below 30 (oversold) | Buy when RSI crosses back above 30 | Stop: $\$0.01$ below recent 2-bar low. Target: RSI $\ge 70$ or stop |

---

## 3. Thomas Bulkowski Swing Trading Strategies (Daily - 1D)

Directory: `strategies/swing/`  
Timeframe: Daily candles (`sleeptime = "1D"`). Uses ATR(14) for volatility-adjusted position sizing and trailing stops.

| Strategy Class | File | Signal Setup | Entry Trigger | Exit & Trailing Stop |
|---|---|---|---|---|
| **`HCRBreakoutLongSwing`** | `hcr_breakout_long.py` | 15-day consolidation range $\le 5\%$ | Buy when `price > highest * 1.002` | Dynamic trailing stop: $1.5 \times \text{ATR}(14)$ below highest price |
| **`HCRBreakoutShortSwing`** | `hcr_breakout_short.py` | 15-day consolidation range $\le 5\%$ | Short when `price < lowest * 0.998` | Dynamic trailing stop: $1.5 \times \text{ATR}(14)$ above lowest price |
| **`GapSetupLongSwing`** | `gap_setup_long.py` | Daily gap down $\ge 2\%$ | Buy fade when `current_price > today_open` | Target: Yesterday's close (gap fill). Stop: $\text{today\_open} - (0.5 \times \text{ATR})$ |
| **`GapSetupShortSwing`** | `gap_setup_short.py` | Daily gap up $\ge 2\%$ | Short fade when `current_price < today_open` | Target: Yesterday's close (gap fill). Stop: $\text{today\_open} + (0.5 \times \text{ATR})$ |
| **`FibRetraceSwing`** | `fib_retrace_swing.py` | 10-day swing range | Buy in 38.2%–61.8% fib zone when today candle is green | Dynamic trailing stop: $2.0 \times \text{ATR}(14)$ |
| **`RetraceLongSwing`** | `retrace_long.py` | 3 consecutive down days | Buy when `current_price > yesterday_high` | Dynamic trailing stop: $1.5 \times \text{ATR}(14)$ |
| **`RetraceShortSwing`** | `retrace_short.py` | 3 consecutive up days | Short when `current_price < yesterday_low` | Dynamic trailing stop: $1.5 \times \text{ATR}(14)$ |
| **`RSISetupSwing`** | `rsi_setup.py` | Daily RSI(14) crosses above 30 from oversold | Buy on crossover | Dynamic trailing stop: $2.0 \times \text{ATR}(14)$. Target: $\text{RSI} \ge 70$ |

---

## 4. Baseplate & Shared Utilities

- **`OpeningGap` (`strat_baseplate.py`)**: Canonical reference template implementing premarket volume threshold filters based on the CBOE Volatility Index ($VIX). Implements 6-day Good-Till-Date (GTD) order expiration logic.
- **`PlottableStrategyMixin` (`plot_mixin.py`)**: Mixin providing a clean contract (`get_plot_spec()`) that decouples strategy code from charting libraries, outputting standalone interactive HTML charts via Plotly.
- **`TradeTracker` & `Trade` (`trade_tracker.py`)**: Comprehensive trade tracking ledger recording entry/exit prices, timestamps, PnL, PnL percentage, durations, winning streaks, and Sharpe/drawdown metrics.
