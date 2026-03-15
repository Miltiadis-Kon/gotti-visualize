"""
Signal data model.

Typed dataclass representing a single parsed trading signal from the cloud CSV export.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    """A single trading signal with parsed fields."""

    signal_id: str
    ticker: str
    signal_date: date
    signal_position: str  # "BUY" or "SELL"

    # Foreign keys to other cloud tables
    calendar_event_keys: List[str] = field(default_factory=list)
    news_keys: List[str] = field(default_factory=list)
    fundamental_analysis_key: Optional[str] = None

    # Parsed JSON fields
    sentiment: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Sentiment convenience properties
    # ------------------------------------------------------------------

    @property
    def gemini_sentiment(self) -> Optional[int]:
        """Gemini sentiment score (1-5 scale) from the sentiment dict."""
        if self.sentiment and "gemini_sentiment" in self.sentiment:
            try:
                return int(self.sentiment["gemini_sentiment"])
            except (ValueError, TypeError):
                return None
        return None

    @property
    def news_sentiment(self) -> Optional[str]:
        """News sentiment label, e.g. 'POSITIVE', 'NEGATIVE'."""
        if self.sentiment:
            return self.sentiment.get("news_sentiment")
        return None

    @property
    def fundamental_sentiment(self) -> Optional[str]:
        """Fundamental analysis sentiment, e.g. 'BUY', 'HOLD', 'SELL'."""
        if self.sentiment:
            return self.sentiment.get("fundamental_sentiment")
        return None

    @property
    def gemini_impact_type(self) -> Optional[str]:
        """Gemini impact classification, e.g. 'High', 'Medium', 'Low'."""
        if self.sentiment:
            return self.sentiment.get("gemini_impact_type")
        return None

    @property
    def gemini_analysis(self) -> Optional[str]:
        """Gemini free-text analysis summary."""
        if self.sentiment:
            return self.sentiment.get("gemini_analysis")
        return None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (JSON-safe types)."""
        return {
            "signal_id": self.signal_id,
            "ticker": self.ticker,
            "signal_date": self.signal_date.isoformat(),
            "signal_position": self.signal_position,
            "calendar_event_keys": self.calendar_event_keys,
            "news_keys": self.news_keys,
            "fundamental_analysis_key": self.fundamental_analysis_key,
            "sentiment": self.sentiment,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"Signal(ticker={self.ticker!r}, date={self.signal_date}, "
            f"position={self.signal_position!r}, id={self.signal_id[:8]}…)"
        )
