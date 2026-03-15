"""
signal_processor — Parse, store, and query trading signals.

Quick start:
    from signal_processor import SignalLoader

    store = SignalLoader.load_from_csv("cloud/signals.csv")
    nvda_buys = store.by_ticker("NVDA").by_position("BUY")
"""

from .loader import SignalLoader
from .models import Signal
from .store import SignalStore

__all__ = ["Signal", "SignalLoader", "SignalStore"]
