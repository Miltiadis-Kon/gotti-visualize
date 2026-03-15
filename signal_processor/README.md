# Signal Processor

A professional Python module for parsing, storing, and querying trading signals from the Gotti cloud CSV exports.

## Overview

The signal processor loads signals from `cloud/signals.csv`, parses all JSON-encoded fields (sentiment, calendar keys, news keys), and provides a **chainable query API** for filtering and grouping signals — ready to feed into your trading strategies.

### Architecture

```
signal_processor/
├── __init__.py    — Package exports (Signal, SignalLoader, SignalStore)
├── models.py      — Signal dataclass with typed fields & sentiment accessors
├── store.py       — SignalStore: chainable filter/group/sort/convert container
├── loader.py      — SignalLoader: CSV parser with JSON & NULL handling
└── README.md      — This file
```

## Quick Start

```python
from signal_processor import SignalLoader

# Load all signals
store = SignalLoader.load_from_csv("cloud/signals.csv")
print(store)  # SignalStore(count=18655, tickers=1842)
```

## Querying Signals

### By Ticker
```python
nvda = store.by_ticker("NVDA")
for signal in nvda:
    print(f"  {signal.signal_date} | {signal.signal_position}")
```

### By Date
```python
from datetime import date

today = store.by_date(date(2026, 2, 12))
feb_signals = store.by_date_range(date(2026, 2, 1), date(2026, 2, 28))
```

### By Position
```python
buys = store.by_position("BUY")
sells = store.by_position("SELL")
```

### Chaining Filters
```python
# NVDA BUY signals in February 2026
nvda_feb_buys = (
    store
    .by_ticker("NVDA")
    .by_position("BUY")
    .by_date_range(date(2026, 2, 1), date(2026, 2, 28))
)
```

### Sentiment Filters
```python
# Signals with high gemini sentiment (4-5)
high_sentiment = store.by_sentiment_score(min_score=4, max_score=5)

# Signals where fundamental analysis says BUY
fund_buys = store.by_fundamental_sentiment("BUY")
```

### Custom Filters
```python
# Any arbitrary filter
has_news = store.apply_filter(lambda s: len(s.news_keys) > 0)
has_calendar = store.apply_filter(lambda s: len(s.calendar_event_keys) > 0)
```

## Grouping

```python
# Group by ticker — returns dict[str, SignalStore]
by_ticker = store.group_by_ticker()
for ticker, signals in by_ticker.items():
    print(f"{ticker}: {signals.count} signals")

# Group by date — returns dict[date, SignalStore]
by_date = store.group_by_date()
```

## Sorting

```python
# Sort by date (oldest first)
chronological = store.sorted_by_date(ascending=True)

# Sort by ticker then date
alphabetical = store.sorted_by_ticker()
```

## Converting to Pandas

```python
df = store.by_ticker("AAPL").to_dataframe()
print(df.columns.tolist())
# ['signal_id', 'ticker', 'signal_date', 'signal_position', ...]
```

## Accessors

```python
store.count               # Number of signals
store.tickers             # Unique sorted ticker list
store.dates               # Unique sorted date list
store.to_list()           # Raw list[Signal]
store[0]                  # First signal
len(store)                # Same as .count
```

## Signal Properties

Each `Signal` object exposes sentiment data as properties:

```python
signal = store.by_ticker("AAPL")[0]
signal.gemini_sentiment          # int (1-5) or None
signal.news_sentiment            # "POSITIVE" / "NEGATIVE" or None
signal.fundamental_sentiment     # "BUY" / "HOLD" / "SELL" or None
signal.gemini_impact_type        # "High" / "Medium" / "Low" or None
signal.gemini_analysis           # Free-text summary or None
```

## Adding Custom Filters (Strategy Integration)

The `apply_filter` method accepts any `Callable[[Signal], bool]`, so you can define
your evaluation strategy as a set of filter functions:

```python
def my_strategy_filter(signal: Signal) -> bool:
    """Only keep signals with high confidence + news backing."""
    return (
        signal.gemini_sentiment is not None
        and signal.gemini_sentiment >= 4
        and len(signal.news_keys) > 0
        and signal.fundamental_sentiment != "SELL"
    )

candidates = store.by_position("BUY").apply_filter(my_strategy_filter)
```

## CSV Format Expected

The loader expects the standard `signals.csv` export with these columns:

| Column | Format |
|--------|--------|
| `signal_id` | String (MD5 hash) |
| `ticker` | Stock symbol |
| `signal_date` | `YYYY-MM-DD` |
| `signal_position` | `BUY` / `SELL` |
| `calendar_event_keys` | JSON array of entry_key strings |
| `news_keys` | JSON array of entry_key strings |
| `fundamental_analysis_key` | String or `\N` for NULL |
| `sentiment` | JSON dict |
| `created_at` | `YYYY-MM-DD HH:MM:SS` |
| `updated_at` | `YYYY-MM-DD HH:MM:SS` |
| `metadata` | JSON dict |
