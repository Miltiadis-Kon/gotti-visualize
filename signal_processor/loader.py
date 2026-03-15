"""
Signal CSV loader.

Reads cloud/signals.csv, handles MySQL-style NULLs (`\\N`), embedded JSON,
and quoting edge cases. Returns a fully parsed SignalStore.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import Signal
from .store import SignalStore


# MySQL CSV export represents NULL as literal \N
_MYSQL_NULL = "\\N"


class SignalLoader:
    """Loads and parses signal data from CSV files."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_csv(path: str) -> SignalStore:
        """
        Load signals from a CSV file and return a SignalStore.

        Parameters
        ----------
        path : str
            Absolute or relative path to the signals CSV.

        Returns
        -------
        SignalStore
            A store containing all successfully parsed signals.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Signals CSV not found: {path}")

        df = pd.read_csv(
            path,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            on_bad_lines="skip",
            dtype=str,  # read everything as strings — we parse below
        )

        signals: List[Signal] = []
        for _, row in df.iterrows():
            try:
                sig = SignalLoader._parse_row(row)
                if sig is not None:
                    signals.append(sig)
            except Exception:
                # Skip rows that fail to parse
                continue

        return SignalStore(signals)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_row(row: pd.Series) -> Optional[Signal]:
        """Parse a single DataFrame row into a Signal dataclass."""
        signal_id = str(row.get("signal_id", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        if not signal_id or not ticker:
            return None

        signal_date = SignalLoader._parse_date(row.get("signal_date"))
        if signal_date is None:
            return None

        return Signal(
            signal_id=signal_id,
            ticker=ticker,
            signal_date=signal_date,
            signal_position=str(row.get("signal_position", "")).strip().upper(),
            calendar_event_keys=SignalLoader._parse_json_list(
                row.get("calendar_event_keys")
            ),
            news_keys=SignalLoader._parse_json_list(row.get("news_keys")),
            fundamental_analysis_key=SignalLoader._nullable_str(
                row.get("fundamental_analysis_key")
            ),
            sentiment=SignalLoader._parse_json_dict(row.get("sentiment")),
            metadata=SignalLoader._parse_json_dict(row.get("metadata")) or {},
            created_at=SignalLoader._parse_datetime(row.get("created_at")),
            updated_at=SignalLoader._parse_datetime(row.get("updated_at")),
        )

    # ------------------------------------------------------------------
    # Type conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nullable_str(val: Any) -> Optional[str]:
        """Return None for MySQL NULL / NaN / empty, else the stripped string."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        if s in ("", _MYSQL_NULL):
            return None
        return s

    @staticmethod
    def _parse_date(val: Any) -> Optional[date]:
        """Parse a date string (YYYY-MM-DD)."""
        s = SignalLoader._nullable_str(val)
        if s is None:
            return None
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_datetime(val: Any) -> Optional[datetime]:
        """Parse a datetime string."""
        s = SignalLoader._nullable_str(val)
        if s is None:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_json_list(val: Any) -> List[str]:
        """Parse a JSON-encoded list of strings; return [] on failure."""
        s = SignalLoader._nullable_str(val)
        if s is None:
            return []
        try:
            result = json.loads(s)
            if isinstance(result, list):
                return [str(x) for x in result]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    @staticmethod
    def _parse_json_dict(val: Any) -> Optional[Dict[str, Any]]:
        """Parse a JSON-encoded dict; return None on failure."""
        s = SignalLoader._nullable_str(val)
        if s is None:
            return None
        try:
            result = json.loads(s)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        return None
