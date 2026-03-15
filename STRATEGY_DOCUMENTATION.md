# Key Levels Trading Strategy — In-Depth Documentation

> Technical documentation for `base_key_levels_strategy.py`, `multi_tf_strategy.py`, and `trade_tracker.py`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Lumibot Lifecycle Methods — How They Apply](#2-lumibot-lifecycle-methods--how-they-apply)
3. [BaseKeyLevelsStrategy — The Abstract Foundation](#3-basekeylevelsstrategy--the-abstract-foundation)
   - 3.1 [Class Definition & Parameters](#31-class-definition--parameters)
   - 3.2 [`initialize()` — One-Time Setup](#32-initialize--one-time-setup)
   - 3.3 [`on_trading_iteration()` — The Main Loop](#33-on_trading_iteration--the-main-loop)
   - 3.4 [`on_filled_order()` — Order Fill Handling](#34-on_filled_order--order-fill-handling)
   - 3.5 [`after_market_closes()` — End-of-Day Persistence](#35-after_market_closes--end-of-day-persistence)
   - 3.6 [`on_abrupt_closing()` — Graceful Shutdown](#36-on_abrupt_closing--graceful-shutdown)
   - 3.7 [Abstract Methods (Template Method Pattern)](#37-abstract-methods-template-method-pattern)
   - 3.8 [Helper Methods](#38-helper-methods)
4. [MultiTimeframeKeyLevelsStrategy — The Concrete Implementation](#4-multitimeframekeylevelsstrategy--the-concrete-implementation)
   - 4.1 [Strategy Parameters](#41-strategy-parameters)
   - 4.2 [`get_entry_signal()` — LONG + SHORT Logic](#42-get_entry_signal--long--short-logic)
   - 4.3 [`_check_long_entry()` — Buying at Support](#43-_check_long_entry--buying-at-support)
   - 4.4 [`_check_short_entry()` — Selling at Resistance](#44-_check_short_entry--selling-at-resistance)
   - 4.5 [`get_exit_signal()` — TP/SL Monitoring](#45-get_exit_signal--tpsl-monitoring)
   - 4.6 [TP/SL Threshold Calculations](#46-tpsl-threshold-calculations)
   - 4.7 [`run_backtest()` — Backtesting Runner](#47-run_backtest--backtesting-runner)
5. [TradeTracker — Trade Recording & Analytics](#5-tradetracker--trade-recording--analytics)
   - 5.1 [Trade Dataclass](#51-trade-dataclass)
   - 5.2 [TradeTracker Class](#52-tradetracker-class)
   - 5.3 [Statistics & Serialization](#53-statistics--serialization)
6. [Execution Flow — End to End](#6-execution-flow--end-to-end)
7. [Key Level Caching Mechanism](#7-key-level-caching-mechanism)
8. [Duplicate Entry Prevention](#8-duplicate-entry-prevention)
9. [Position Sizing Formula](#9-position-sizing-formula)

---

## 1. Architecture Overview

The strategy system follows the **Template Method design pattern** (GoF). There are two layers:

```
┌─────────────────────────────────────────┐
│        lumibot.strategies.Strategy       │  ← Lumibot framework base class
├─────────────────────────────────────────┤
│       BaseKeyLevelsStrategy (ABC)        │  ← Our abstract base class
│  - initialize()                          │     Implements all lifecycle hooks
│  - on_trading_iteration()                │     Delegates entry/exit to child
│  - on_filled_order()                     │
│  - after_market_closes()                 │
│  - on_abrupt_closing()                   │
│  - Position sizing, caching, tracking    │
├─────────────────────────────────────────┤
│    MultiTimeframeKeyLevelsStrategy       │  ← Concrete child strategy
│  - get_entry_signal()  (LONG + SHORT)    │     Only implements WHAT to trade
│  - get_exit_signal()   (TP/SL check)     │
│  - TP/SL threshold calculations          │
└─────────────────────────────────────────┘
```

**Why this separation?**
- The **base class** owns all infrastructure: trade tracking, order management, key level loading/caching, position sizing, file I/O, duplicate prevention.
- The **child class** only decides WHEN to enter and WHEN to exit. This makes it trivial to create new strategies — just implement two methods.

Supporting module:
- **`trade_tracker.py`** — A standalone `Trade` dataclass + `TradeTracker` class for recording, serializing, and analyzing trades.

---

## 2. Lumibot Lifecycle Methods — How They Apply

Lumibot's `Strategy` base class defines a **React.js-inspired lifecycle** ([Lumibot Docs](https://lumibot.lumiwealth.com/lifecycle_methods.html)). The framework calls these methods at specific points during a trading session:

| Lifecycle Method | When Called | Our Implementation |
|---|---|---|
| `initialize()` | **Once**, before any trading begins | Set `sleeptime`, create `TradeTracker`, load parameters, generate output filename |
| `on_trading_iteration()` | **Every `sleeptime` interval** while market is open | Load/refresh key levels, get current price, delegate to entry/exit logic |
| `on_filled_order()` | **Each time** a submitted order is filled by the broker | Log fill details, record trade in `TradeTracker`, calculate PnL on sell |
| `after_market_closes()` | **Once per day**, after closing bell | Save trades (JSON + CSV) and levels history to `logs/` directory |
| `on_abrupt_closing()` | When strategy is **interrupted** (Ctrl+C, crash, end of backtest) | Close any open positions, save all data, print trade summary |

**Execution order in a single trading day:**
```
initialize()  →  [market opens]  →  on_trading_iteration()  →  sleep 5min
                                  →  on_trading_iteration()  →  sleep 5min
                                  →  on_trading_iteration()  →  sleep 5min
                                  →  ... (repeats all day)
                                  →  on_filled_order()       (triggered by fills)
                  [market closes] →  after_market_closes()
```

If the bot is interrupted at any point: `on_abrupt_closing()` fires.

---

## 3. BaseKeyLevelsStrategy — The Abstract Foundation

**File:** `strategies/base_key_levels_strategy.py` (~600 lines)

### 3.1 Class Definition & Parameters

```python
class BaseKeyLevelsStrategy(Strategy):
```

Extends Lumibot's `Strategy` class. Defines default parameters that child classes can override:

| Parameter | Default | Purpose |
|---|---|---|
| `Ticker` | `Asset("NVDA", STOCK)` | The asset to trade |
| `RISK_PERCENT` | `0.02` (2%) | Percentage of portfolio risked per trade |
| `MIN_IMPORTANCE` | `1` | Minimum importance score for key levels (higher = appears on more timeframes) |
| `TIMEFRAMES` | `['1d','4h','1h','15m']` | Timeframes to scan for support/resistance |
| `RECALC_FREQUENCY` | `"daily"` | How often to recalculate levels: `"daily"`, `"weekly"`, or `"once"` |
| `ENTRY_THRESHOLD` | `0.05` (5%) | Price must be within this % of a level to trigger entry |
| `EXIT_THRESHOLD` | `0.1` (10%) | Tolerance for TP/SL matching on manual exit check |

**Class-level variables** (shared across all instances, persist during backtesting):
- `_cached_levels = {}` — Dictionary caching computed key levels to avoid redundant recalculation.
- `_last_trade_tracker = None` — Stores the final `TradeTracker` after backtest completes, allowing post-backtest analysis.

### 3.2 `initialize()` — One-Time Setup

**Lumibot reference:** Called once before any trading begins. Use it to set `self.sleeptime` and initialize instance variables. ([Docs](https://lumibot.lumiwealth.com/lifecycle_methods.initialize.html))

What happens step by step:

1. **Set `sleeptime = "5M"`** — The strategy's `on_trading_iteration()` runs every 5 minutes.
2. **Load all parameters** from `self.parameters` into instance variables (risk_percent, min_importance, timeframes, etc.).
3. **Initialize level storage** — `merged_levels`, `support_levels`, `resistance_levels` all start as `None`.
4. **Initialize trade tracking variables** — `entry_support`, `target_resistance`, `last_levels_date`, `current_trade_id`.
5. **Create a `TradeTracker`** instance for the ticker.
6. **Generate the output filename** using the **real system clock** (not simulation time):
   ```
   Format: {DDMM}_{HHMM}_{TICKER}_{STRATEGY}
   Example: 2412_1738_NVDA_multitfkeylevels
   ```
   This ensures all trades from a single backtest run are saved to one file, regardless of how many simulated days are processed.
7. **Initialize `_levels_history`** — A list that accumulates all calculated levels per day for later visualization.
8. **Initialize `_entered_levels`** — A `set()` of `(price, trade_type)` tuples for duplicate entry prevention.
9. **Call `self.on_strategy_start()`** — A hook for child classes to perform custom initialization.

### 3.3 `on_trading_iteration()` — The Main Loop

**Lumibot reference:** Called every `self.sleeptime` interval while the market is open. Contains the core trading logic. ([Docs](https://lumibot.lumiwealth.com/lifecycle_methods.on_trading_iteration.html))

```
on_trading_iteration()
├── 1. Check if levels need reloading → _should_reload_levels()
│   └── If yes → _load_key_levels()
├── 2. Get current price via self.get_last_price()
│   └── If None → return (no data yet)
├── 3. Get current position via self.get_position()
├── 4. ALWAYS check for entry → _handle_entry()
│   └── Calls child's get_entry_signal()
└── 5. If position exists, check for exit → _handle_exit()
    └── Calls child's get_exit_signal()
```

**Key design decisions:**
- Entry is checked **every iteration**, even when a position exists. The `_entered_levels` set prevents duplicate entries at the same level — but allows entering at a *different* level if conditions arise.
- Level reloading is controlled by `RECALC_FREQUENCY`. For `"daily"`, levels recalculate when the date changes. For `"weekly"`, every 7 days or on Monday. For `"once"`, levels are calculated once and never refreshed.

### 3.4 `on_filled_order()` — Order Fill Handling

**Lumibot reference:** Triggered each time a submitted order is filled by the broker. Receives the position, order, filled price, quantity, and multiplier. ([Docs](https://lumibot.lumiwealth.com/lifecycle_methods.on_filled_order.html))

**On BUY fill:**
- Logs the fill with price and quantity.
- Logs the support/resistance levels that triggered the trade.

**On SELL fill:**
- Logs the fill.
- Calls `trade_tracker.close_trade()` with the filled price.
- Determines exit reason via `_determine_exit_reason()`:
  - If exit price ≥ 99% of TP → `"TP"` (take profit)
  - If exit price ≤ 101% of SL → `"SL"` (stop loss)
  - Otherwise → `"MANUAL"`
- Logs the PnL result.
- Resets `current_trade_id`, `entry_support`, `target_resistance`.

### 3.5 `after_market_closes()` — End-of-Day Persistence

**Lumibot reference:** Executes once after the market closes each day. Commonly used for dumping stats and reports. ([Docs](https://lumibot.lumiwealth.com/lifecycle_methods.after_market_closes.html))

Three files are saved to `logs/`:

| File | Format | Contents |
|---|---|---|
| `{filename}.json` | JSON | All trades with full details + statistics |
| `{filename}.csv` | CSV | Tabular trade data (DataFrame export) |
| `{filename}_levels.json` | JSON | Historical key levels for each day |

The `TradeTracker` instance is also stored in the class-level `_last_trade_tracker` variable so it can be accessed after a backtest completes.

### 3.6 `on_abrupt_closing()` — Graceful Shutdown

**Lumibot reference:** Fires when the strategy is interrupted (keyboard interrupt, crash). Used for graceful cleanup. ([Docs](https://lumibot.lumiwealth.com/lifecycle_methods.on_abrupt_closing.html))

What happens:
1. **Close any open position** at the current market price.
2. **Record the close** in `TradeTracker` with exit reason `"END_SIM"`.
3. **Call `after_market_closes()`** to save all data files.
4. **Print the trade summary** via `trade_tracker.print_summary()`.

The `_close_all_positions()` helper handles both LONG and SHORT positions:
- LONG (positive quantity) → `self.sell_all()`
- SHORT (negative quantity) → Create a BUY order for `abs(quantity)` shares

### 3.7 Abstract Methods (Template Method Pattern)

These two methods **must** be implemented by any child class:

#### `get_entry_signal(current_price, support_levels, resistance_levels) → Optional[Dict]`

Receives the current price and DataFrames of support/resistance levels. Must return either:
- `None` — No entry signal
- A dictionary with keys:
  ```python
  {
      'trade_type': 'BUY' or 'SELL',
      'entry_price': float,
      'take_profit': float,
      'stop_loss': float,
      'support_level': float,
      'resistance_level': float,
      'quantity': int  # Optional — auto-calculated if omitted
  }
  ```

#### `get_exit_signal(current_price, position, entry_support, target_resistance) → Optional[str]`

Receives the current price, position object, and the support/resistance levels where the trade was entered. Must return either:
- `None` — Hold the position
- An exit reason string: `"TP"`, `"SL"`, or `"MANUAL"`

This acts as a **backup** to the broker's bracket order (which handles TP/SL automatically). It catches cases where the bracket order hasn't fired but the price has met the exit criteria.

### 3.8 Helper Methods

| Method | Purpose |
|---|---|
| `_handle_entry(price)` | Orchestrates entry: calls `get_entry_signal()`, checks duplicates, sizes position, records trade, submits order |
| `_handle_exit(price, position)` | Orchestrates exit: calls `get_exit_signal()`, closes position if triggered |
| `_close_all_positions()` | Closes all positions, handling both LONG (sell_all) and SHORT (buy to cover) |
| `_determine_exit_reason(price)` | Heuristic: compares exit price to TP/SL to classify the exit |
| `get_position_sizing(entry, sl, type)` | Risk-based position sizing (see [Section 9](#9-position-sizing-formula)) |
| `_should_reload_levels(date)` | Determines if key levels need recalculation based on frequency |
| `_load_key_levels(datetime)` | Fetches levels via `KeyLevels` class, filters by importance, caches results |
| `_is_level_already_entered(price, type)` | Checks `_entered_levels` set for duplicates (see [Section 8](#8-duplicate-entry-prevention)) |
| `_mark_level_entered(price, type)` | Adds level to `_entered_levels` set |
| `_log_levels_to_history(date)` | Appends current levels snapshot to `_levels_history` |
| `_save_levels_history(path)` | Serializes `_levels_history` to JSON |
| `on_strategy_start()` | Hook for child initialization (no-op in base) |
| `get_strategy_name()` | Returns class name by default; override for custom names |

---

## 4. MultiTimeframeKeyLevelsStrategy — The Concrete Implementation

**File:** `strategies/multi_tf_strategy.py` (~300 lines)

This is a **production-ready** strategy that trades LONG (BUY) at support and SHORT (SELL) at resistance, using multi-timeframe key levels with importance scoring and configurable risk-reward filtering.

### 4.1 Strategy Parameters

Overrides the base class defaults:

| Parameter | Value | Difference from Base |
|---|---|---|
| `MIN_IMPORTANCE` | `3` | Requires at least 1H-level importance (base: 1) |
| `TIMEFRAMES` | `['1d','4h','1h','15m','5m']` | Adds 5m timeframe |
| `PRICE_THRESHOLD` | `0.5` | Merging threshold for nearby levels |
| `ENTRY_THRESHOLD` | `0.005` (0.5%) | Much tighter than base's 5% — price must be very close to level |
| `EXIT_THRESHOLD` | `0.01` (1%) | Tighter than base's 10% |
| `TP_THRESHOLD` | `0.02` (2%) | Take profit 2% before the target level |
| `SL_THRESHOLD` | `0.05` (5%) | Stop loss 5% beyond the entry level |
| `MIN_RISK_REWARD` | `1.5` | Minimum 1.5:1 reward-to-risk ratio |

### 4.2 `get_entry_signal()` — LONG + SHORT Logic

The entry logic is **bidirectional**:

```
get_entry_signal()
├── Try LONG entry → _check_long_entry()
│   └── If signal found → return it
└── Try SHORT entry → _check_short_entry()
    └── If signal found → return it
```

LONG is checked **first**, giving it priority. If a LONG signal exists at support AND a SHORT signal exists at resistance simultaneously, the LONG signal wins.

### 4.3 `_check_long_entry()` — Buying at Support

**Algorithm:**

1. **Sort support levels** by importance (descending) then distance to current price (ascending). This prioritizes stronger, closer supports.
2. **For each support level:**
   - Check if current price is within `±0.5%` of the support price (entry threshold).
   - Calculate the stop loss: `SL = support_price × (1 - 0.05)` — 5% below support.
   - Calculate `risk_per_share = entry_price - stop_loss`. Skip if ≤ 0.
3. **For each resistance level above the entry price:**
   - Calculate take profit: `TP = resistance_price × (1 - 0.02)` — 2% before resistance.
   - Calculate `reward_per_share = take_profit - entry_price`.
   - Calculate `R:R = reward / risk`.
   - If R:R ≥ 1.5 → **Signal found**. Return the trade parameters.
4. **Fallback:** If no resistance meets the R:R requirement but resistances exist above, use a **synthetic TP** calculated as `entry + (risk × min_R:R)`. This guarantees the minimum R:R ratio is always met.

### 4.4 `_check_short_entry()` — Selling at Resistance

**Mirror logic of the LONG entry:**

1. **Sort resistance levels** by importance (desc) then distance (asc).
2. **For each resistance level:**
   - Check if price is within ±0.5% of resistance.
   - Calculate SL: `SL = resistance_price × (1 + 0.05)` — 5% above resistance.
   - Calculate `risk_per_share = stop_loss - entry_price`.
3. **For each support level below:**
   - Calculate TP: `TP = support_price × (1 + 0.02)` — 2% above support.
   - Calculate `reward_per_share = entry_price - take_profit`.
   - If R:R ≥ 1.5 → **Signal found**.
4. **Fallback:** Synthetic TP at `entry - (risk × min_R:R)`.

### 4.5 `get_exit_signal()` — TP/SL Monitoring

Acts as a **backup exit mechanism** in case the broker's bracket order hasn't triggered.

Retrieves the current trade from `TradeTracker` and checks:

**For LONG positions:**
- `current_price >= TP - (TP × 1%)` → Return `"TP"`
- `current_price <= SL + (SL × 1%)` → Return `"SL"`

**For SHORT positions:**
- `current_price <= TP + (TP × 1%)` → Return `"TP"`
- `current_price >= SL - (SL × 1%)` → Return `"SL"`

The `exit_threshold` (1%) creates a **tolerance zone** around TP/SL, triggering the manual exit slightly before the exact level. This accounts for slippage, spread, and cases where the bracket order is slow.

### 4.6 TP/SL Threshold Calculations

| Calculation | Formula | Example (price=$100) |
|---|---|---|
| Long TP | `resistance × (1 - 0.02)` | Resistance at $110 → TP = $107.80 |
| Long SL | `support × (1 - 0.05)` | Support at $95 → SL = $90.25 |
| Short TP | `support × (1 + 0.02)` | Support at $90 → TP = $91.80 |
| Short SL | `resistance × (1 + 0.05)` | Resistance at $105 → SL = $110.25 |

**Rationale for asymmetric thresholds:**
- TP at 2% before the level: Takes profit before the level is reached, since levels often act as reversal zones.
- SL at 5% beyond the level: Gives the trade breathing room to survive false breakouts.

### 4.7 `run_backtest()` — Backtesting Runner

A convenience function at module level:

```python
run_backtest(
    ticker="NVDA",
    start_date=None,        # Defaults to 60 days ago
    end_date=None,           # Defaults to today
    budget=10000,
    min_importance=3,
    min_risk_reward=1.5
)
```

Uses `YahooDataBacktesting` as the data source. Parameters are passed through to the strategy's `parameters` dictionary.

The `if __name__ == "__main__"` block runs a default backtest on PLTR for December 2025.

---

## 5. TradeTracker — Trade Recording & Analytics

**File:** `strategies/trade_tracker.py` (~364 lines)

### 5.1 Trade Dataclass

The `Trade` dataclass stores every detail of a single trade:

| Field | Type | Description |
|---|---|---|
| `trade_id` | `int` | Auto-incrementing unique ID |
| `ticker` | `str` | Asset symbol |
| `trade_type` | `str` | `"BUY"` (long) or `"SELL"` (short) |
| `date_executed` | `datetime` | Entry timestamp |
| `entry_price` | `float` | Fill price at entry |
| `quantity` | `int` | Number of shares |
| `take_profit` | `float` | TP price level |
| `stop_loss` | `float` | SL price level |
| `support_level` | `float` | Support that triggered the trade |
| `resistance_level` | `float` | Target resistance (or entry for shorts) |
| `date_completed` | `datetime` | Exit timestamp (None if open) |
| `exit_price` | `float` | Fill price at exit |
| `exit_reason` | `str` | `"TP"`, `"SL"`, `"MANUAL"`, `"EOD"`, `"END_SIM"` |
| `pnl` | `float` | Dollar profit/loss |
| `pnl_percent` | `float` | Percentage return |

**Computed properties:**
- `is_open` → `True` if `date_completed is None`
- `is_winner` → `True` if `pnl > 0`
- `duration` → Hours between entry and exit (or `None` if open)

Serialization: `to_dict()` / `from_dict()` with ISO 8601 datetime handling.

### 5.2 TradeTracker Class

| Method | Description |
|---|---|
| `open_trade(...)` | Creates a `Trade` record, assigns an auto-incrementing ID, returns the ID |
| `close_trade(trade_id, date, exit_price, reason)` | Sets exit fields, calculates PnL (LONG: `(exit - entry) × qty`, SHORT: `(entry - exit) × qty`), updates running total |
| `get_trade(trade_id)` | Lookup by ID |
| `get_open_trade()` | Returns the first open trade (if any) |

### 5.3 Statistics & Serialization

**`get_statistics()`** returns:
- Total/winning/losing trade counts
- Win rate (%)
- Total PnL ($)
- Average win/loss amounts
- Largest win/loss
- Average trade duration (hours)

**Output formats:**
- `save_to_json(path)` — Full trade data + statistics in JSON
- `save_to_csv(path)` — DataFrame format via `to_dataframe()`
- `load_from_json(path)` — Class method to reload from JSON
- `print_summary()` — Formatted console output with trade stats

---

## 6. Execution Flow — End to End

A complete backtest run:

```
1. MultiTimeframeKeyLevelsStrategy.backtest(YahooDataBacktesting, ...)
   │
2. initialize()
   ├── sleeptime = "5M"
   ├── Create TradeTracker("NVDA")
   ├── Generate filename: "2412_1738_NVDA_multitfkeylevels"
   ├── Initialize _entered_levels = set()
   └── on_strategy_start() → load TP/SL/R:R thresholds, log config
   │
3. [Day 1 - Market Opens]
   │
4. on_trading_iteration() #1
   ├── _should_reload_levels() → True (first time)
   ├── _load_key_levels()
   │   ├── Check cache → miss
   │   ├── KeyLevels("NVDA", as_of_date=...)
   │   ├── find_all_key_levels(timeframes=['1d','4h','1h','15m','5m'])
   │   ├── get_merged_levels() → DataFrame with level_price, importance, type
   │   ├── Filter: importance >= 3
   │   ├── Split into support_levels / resistance_levels
   │   ├── Cache result
   │   └── Log to _levels_history
   ├── get_last_price("NVDA") → $138.50
   ├── get_position("NVDA") → None
   └── _handle_entry($138.50)
       ├── get_entry_signal($138.50, supports, resistances)
       │   ├── _check_long_entry()
       │   │   ├── Sort supports by importance desc, distance asc
       │   │   ├── Support $138.20 (imp=4): within 0.5%? |138.50-138.20|/138.20 = 0.22% ✓
       │   │   ├── SL = 138.20 × 0.95 = $131.29
       │   │   ├── Risk = 138.50 - 131.29 = $7.21/share
       │   │   ├── Resistance $150.00 (above entry): TP = 150 × 0.98 = $147.00
       │   │   ├── Reward = 147.00 - 138.50 = $8.50
       │   │   ├── R:R = 8.50/7.21 = 1.18 < 1.5 ✗
       │   │   ├── Next resistance $160.00: TP = 160 × 0.98 = $156.80
       │   │   ├── Reward = 156.80 - 138.50 = $18.30
       │   │   ├── R:R = 18.30/7.21 = 2.54 ≥ 1.5 ✓
       │   │   └── Return signal {BUY, entry=138.50, TP=156.80, SL=131.29, ...}
       │   └── (SHORT check skipped — LONG found)
       ├── _is_level_already_entered($138.20, "BUY") → False
       ├── get_position_sizing($138.50, $131.29, "BUY")
       │   ├── Portfolio = $10,000
       │   ├── Risk amount = $10,000 × 0.02 = $200
       │   ├── Risk/share = $7.21
       │   ├── Quantity = 200 / 7.21 = 27 shares
       │   └── Max affordable = 10000 × 0.95 / 138.50 = 68 → min(27, 68) = 27
       ├── _mark_level_entered($138.20, "BUY")
       ├── trade_tracker.open_trade(...) → trade_id = 1
       └── submit_order(BUY 27 NVDA, TP=$156.80, SL=$131.29)
   │
5. on_filled_order(position, order, $138.50, 27, 1)
   └── Log: "BUY filled: 27 @ $138.50, Support: $138.20, Target: $160.00"
   │
6. on_trading_iteration() #2...N (every 5 min)
   ├── _should_reload_levels() → False (same day)
   ├── get_last_price() → varies
   ├── _handle_entry() → skipped (level already entered)
   └── _handle_exit()
       └── get_exit_signal() → None (price between SL and TP)
   │
7. ... Price reaches $156.80 → Bracket order fills TP ...
   │
8. on_filled_order(position, order, $156.80, 27, 1)
   ├── trade_tracker.close_trade(id=1, price=$156.80, reason="TP")
   │   └── PnL = (156.80 - 138.50) × 27 = +$494.10
   └── Reset: current_trade_id = None, entry_support = None
   │
9. [Market Closes]
   │
10. after_market_closes()
    ├── Save logs/2412_1738_NVDA_multitfkeylevels.json
    ├── Save logs/2412_1738_NVDA_multitfkeylevels.csv
    └── Save logs/2412_1738_NVDA_multitfkeylevels_levels.json
```

---

## 7. Key Level Caching Mechanism

Key levels are expensive to compute (multiple timeframes, historical data fetch). The caching system avoids redundant computation.

**Cache structure:**
```python
BaseKeyLevelsStrategy._cached_levels = {
    "NVDA_2025-01-15_daily": {
        'merged': DataFrame,
        'support': DataFrame,
        'resistance': DataFrame
    },
    "NVDA_2025-01-16_daily": { ... },
    ...
}
```

**Cache key format:** `{ticker}_{date}_{recalc_frequency}`

**Important:** The cache is a **class-level** variable (`_cached_levels = {}`), which means:
- It persists across trading days within a single backtest run.
- Multiple instances of the strategy share the same cache.
- Prevents re-fetching levels for the same date even if the strategy restarts within the same Python process.

**Reload logic (`_should_reload_levels`):**
| Frequency | Reloads When |
|---|---|
| `"daily"` | Date changes (new trading day) |
| `"weekly"` | 7+ days passed or it's Monday |
| `"once"` | Never after first load |

---

## 8. Duplicate Entry Prevention

The strategy uses `_entered_levels` — a `set` of `(level_price, trade_type)` tuples — to prevent entering the same level multiple times.

**How it works:**
1. Before entering a trade, `_is_level_already_entered()` checks if any previously entered level is within the `ENTRY_THRESHOLD` (0.5%) of the candidate level AND has the same trade type (BUY/SELL).
2. If found → **skip the entry**.
3. If not found → **proceed** and call `_mark_level_entered()` to add it to the set.

**When levels are cleared:**
- `_clear_entered_level()` exists but is NOT called on individual trade closes (by design — once you've entered at a level, you don't re-enter the same day).
- The set naturally resets when a new `_load_key_levels()` call generates different levels (different day = different prices and levels).

**Why this matters:** Without this, the strategy would re-enter a trade at the same support every 5 minutes (each iteration), compounding risk.

---

## 9. Position Sizing Formula

The strategy uses **fixed-percentage risk** position sizing:

$$\text{Quantity} = \frac{\text{Portfolio Value} \times \text{Risk\%}}{|\text{Entry Price} - \text{Stop Loss}|}$$

**With a safety cap:**

$$\text{Max Affordable} = \frac{\text{Portfolio Value} \times 0.95}{\text{Entry Price}}$$

$$\text{Final Quantity} = \min(\text{Quantity}, \text{Max Affordable})$$

**Example:**
- Portfolio: $10,000
- Risk%: 2%
- Entry: $138.50
- Stop Loss: $131.29
- Risk Amount = $10,000 × 0.02 = **$200**
- Risk/Share = |$138.50 − $131.29| = **$7.21**
- Quantity = $200 / $7.21 = **27 shares**
- Max Affordable = ($10,000 × 0.95) / $138.50 = **68 shares**
- Final = min(27, 68) = **27 shares**

This ensures:
1. You never risk more than 2% of portfolio on any single trade.
2. You never use more than 95% of portfolio value (5% cash buffer).
3. Works identically for both LONG and SHORT trades (uses `abs()` on the risk/share calculation).

---

*Generated from source code analysis of `strategies/base_key_levels_strategy.py`, `strategies/multi_tf_strategy.py`, and `strategies/trade_tracker.py`. Lumibot lifecycle references from [lumibot.lumiwealth.com](https://lumibot.lumiwealth.com/lifecycle_methods.html).*
