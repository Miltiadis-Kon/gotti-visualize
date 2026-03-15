"""
Signal store — the main query interface for trading signals.

Holds a list of Signal objects and provides chainable filter/query methods.
Every filter method returns a *new* SignalStore so the original stays untouched.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Dict, List, Optional

import pandas as pd

from .models import Signal


class SignalStore:
    """Immutable, chainable container for querying signals."""

    def __init__(self, signals: Optional[List[Signal]] = None) -> None:
        self._signals: List[Signal] = list(signals) if signals else []

    # ------------------------------------------------------------------
    # Filter methods (all return a new SignalStore)
    # ------------------------------------------------------------------

    def by_ticker(self, ticker: str) -> SignalStore:
        """Filter signals for a specific ticker (case‑insensitive)."""
        t = ticker.upper()
        return SignalStore([s for s in self._signals if s.ticker.upper() == t])

    def by_date(self, target_date: date) -> SignalStore:
        """Filter signals for an exact date."""
        return SignalStore([s for s in self._signals if s.signal_date == target_date])

    def by_date_range(self, start: date, end: date) -> SignalStore:
        """Filter signals within a date range (inclusive on both ends)."""
        return SignalStore(
            [s for s in self._signals if start <= s.signal_date <= end]
        )

    def by_position(self, position: str) -> SignalStore:
        """Filter by signal position, e.g. 'BUY' or 'SELL'."""
        p = position.upper()
        return SignalStore(
            [s for s in self._signals if s.signal_position.upper() == p]
        )

    def by_sentiment_score(
        self, min_score: int = 1, max_score: int = 5
    ) -> SignalStore:
        """Filter by gemini_sentiment score (1‑5 inclusive)."""
        return SignalStore(
            [
                s
                for s in self._signals
                if s.gemini_sentiment is not None
                and min_score <= s.gemini_sentiment <= max_score
            ]
        )

    def by_fundamental_sentiment(self, sentiment: str) -> SignalStore:
        """Filter by fundamental sentiment label (BUY / HOLD / SELL)."""
        sent = sentiment.upper()
        return SignalStore(
            [
                s
                for s in self._signals
                if s.fundamental_sentiment is not None
                and s.fundamental_sentiment.upper() == sent
            ]
        )

    def apply_filter(self, filter_fn: Callable[[Signal], bool]) -> SignalStore:
        """Apply an arbitrary predicate — ultimate extensibility hook."""
        return SignalStore([s for s in self._signals if filter_fn(s)])

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of signals in the store."""
        return len(self._signals)

    @property
    def tickers(self) -> List[str]:
        """Unique, sorted list of ticker symbols."""
        return sorted({s.ticker for s in self._signals})

    @property
    def dates(self) -> List[date]:
        """Unique, sorted list of signal dates (oldest → newest)."""
        return sorted({s.signal_date for s in self._signals})

    def to_list(self) -> List[Signal]:
        """Return the raw list of Signal objects."""
        return list(self._signals)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert signals to a pandas DataFrame."""
        if not self._signals:
            return pd.DataFrame()
        rows = [s.to_dict() for s in self._signals]
        df = pd.DataFrame(rows)
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        return df

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def group_by_ticker(self) -> Dict[str, SignalStore]:
        """Group signals by ticker → {ticker: SignalStore}."""
        groups: Dict[str, List[Signal]] = {}
        for s in self._signals:
            groups.setdefault(s.ticker, []).append(s)
        return {t: SignalStore(sigs) for t, sigs in sorted(groups.items())}

    def group_by_date(self) -> Dict[date, SignalStore]:
        """Group signals by date → {date: SignalStore}."""
        groups: Dict[date, List[Signal]] = {}
        for s in self._signals:
            groups.setdefault(s.signal_date, []).append(s)
        return {d: SignalStore(sigs) for d, sigs in sorted(groups.items())}

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def sorted_by_date(self, ascending: bool = True) -> SignalStore:
        """Return a new store sorted by signal_date."""
        return SignalStore(
            sorted(self._signals, key=lambda s: s.signal_date, reverse=not ascending)
        )

    def sorted_by_ticker(self, ascending: bool = True) -> SignalStore:
        """Return a new store sorted by ticker then date."""
        return SignalStore(
            sorted(
                self._signals,
                key=lambda s: (s.ticker, s.signal_date),
                reverse=not ascending,
            )
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self._signals)

    def __getitem__(self, index):
        return self._signals[index]

    def __bool__(self) -> bool:
        return self.count > 0

    def __repr__(self) -> str:
        return f"SignalStore(count={self.count}, tickers={len(self.tickers)})"
