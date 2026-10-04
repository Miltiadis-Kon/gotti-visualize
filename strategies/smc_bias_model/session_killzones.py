"""
SMC Bias Model: Session Killzones & Liquidation Sweeps (session_killzones.py)
=============================================================================
Rules 2 & 3 of the 5-Rule SMC Intraday Bias Model:
- Rule 2 (Time and Price):
  Execution is restricted strictly to high-probability institutional macro killzones (in Eastern Standard Time):
    * London Killzone: 02:00 AM – 05:00 AM EST (07:00 – 10:00 UTC)
    * New York Killzone: 07:00 AM – 10:00 AM EST (12:00 – 15:00 UTC)
      (extended to 11:30 AM EST for US Equities opening momentum)
  * Hard Mandate: Setups outside the killzone cannot be taken.

- Rule 3 (Liquidation - The Trap):
  Price must purge retail liquidity from prior session benchmark extremes before entering:
    * London Shorts: Must sweep Asian High (ASH) or Frankfurt High.
    * London Longs: Must sweep Asian Low (ASL) or Frankfurt Low.
    * New York Shorts: Must sweep London High (LNH) or Premarket High (PMH).
    * New York Longs: Must sweep London Low (LNL) or Premarket Low (PML).
  * Hard Filter: No liquidation sweep = No trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import pytz


@dataclass
class SessionExtremes:
    high: float
    low: float
    high_time: Any
    low_time: Any
    session_name: str


@dataclass
class LiquidationStatus:
    swept: bool
    sweep_type: str  # "HIGH_SWEPT" or "LOW_SWEPT"
    benchmark_level: float
    benchmark_name: str
    sweep_time: Any


class SessionKillzoneManager:
    """
    Manages session boundaries, tracks session extremes (Asian, London, Premarket),
    and enforces the strict Killzone Gate and Liquidation Sweep rules.
    """

    def __init__(self, allow_ny_extension: bool = True):
        self.tz_est = pytz.timezone("America/New_York")
        self.allow_ny_extension = allow_ny_extension

    def get_est_time(self, dt: Any) -> datetime:
        """Converts a timestamp or datetime to Eastern Time (EST/EDT)."""
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if dt.tzinfo is None:
            # Assume UTC if naive
            dt = pytz.utc.localize(dt)
        return dt.astimezone(self.tz_est)

    def is_in_killzone(self, dt: Any) -> Tuple[bool, str]:
        """
        Rule 2 Check: Checks if the current bar falls within London or NY Killzone.
        Returns: (is_valid, session_name)
        """
        est_dt = self.get_est_time(dt)
        t = est_dt.time()

        # London Killzone: 02:00 - 05:00 EST
        if time(2, 0) <= t < time(5, 0):
            return True, "LONDON_KILLZONE"

        # New York Killzone: 07:00 - 10:00 EST (or 11:30 EST for equities)
        ny_end = time(11, 30) if self.allow_ny_extension else time(10, 0)
        if time(7, 0) <= t < ny_end:
            return True, "NEW_YORK_KILLZONE"

        return False, "OUTSIDE_KILLZONE"

    def extract_prior_session_benchmarks(self, df_bars: pd.DataFrame, current_time: Any) -> Dict[str, SessionExtremes]:
        """
        Extracts benchmark extremes from the lookback window:
        - Asian Session (19:00 prior day to 02:00 EST)
        - London Session (02:00 EST to 07:00 EST)
        - Premarket (04:00 EST to 09:30 EST)
        - Previous Day High / Low
        """
        benchmarks: Dict[str, SessionExtremes] = {}
        if df_bars.empty:
            return benchmarks

        cur_est = self.get_est_time(current_time)
        cur_date = cur_est.date()

        # Build EST series
        est_times = [self.get_est_time(t) for t in df_bars.index]
        df_est = df_bars.copy()
        df_est["est_time"] = est_times
        df_est["est_date"] = [t.date() for t in est_times]
        df_est["time_only"] = [t.time() for t in est_times]

        # 1. Asian Session (19:00 prev day to 02:00 current day)
        # Find bars between 19:00 on cur_date - 1 day to 02:00 on cur_date
        asian_mask = (
            ((df_est["est_date"] < cur_date) & (df_est["time_only"] >= time(19, 0))) |
            ((df_est["est_date"] == cur_date) & (df_est["time_only"] < time(2, 0)))
        )
        asian_bars = df_est[asian_mask]
        if not asian_bars.empty:
            benchmarks["ASIAN"] = SessionExtremes(
                high=float(asian_bars["high"].max()),
                low=float(asian_bars["low"].min()),
                high_time=asian_bars["high"].idxmax(),
                low_time=asian_bars["low"].idxmin(),
                session_name="ASIAN_SESSION"
            )

        # 2. London Session (02:00 to 07:00 EST on current day)
        london_mask = (df_est["est_date"] == cur_date) & (df_est["time_only"] >= time(2, 0)) & (df_est["time_only"] < time(7, 0))
        london_bars = df_est[london_mask]
        if not london_bars.empty:
            benchmarks["LONDON"] = SessionExtremes(
                high=float(london_bars["high"].max()),
                low=float(london_bars["low"].min()),
                high_time=london_bars["high"].idxmax(),
                low_time=london_bars["low"].idxmin(),
                session_name="LONDON_SESSION"
            )

        # 3. Premarket Session (04:00 to 09:30 EST on current day)
        pm_mask = (df_est["est_date"] == cur_date) & (df_est["time_only"] >= time(4, 0)) & (df_est["time_only"] < time(9, 30))
        pm_bars = df_est[pm_mask]
        if not pm_bars.empty:
            benchmarks["PREMARKET"] = SessionExtremes(
                high=float(pm_bars["high"].max()),
                low=float(pm_bars["low"].min()),
                high_time=pm_bars["high"].idxmax(),
                low_time=pm_bars["low"].idxmin(),
                session_name="PREMARKET"
            )

        # 4. Previous Day Full RTH
        prior_dates = [d for d in set(df_est["est_date"]) if d < cur_date]
        if prior_dates:
            last_date = max(prior_dates)
            prior_bars = df_est[df_est["est_date"] == last_date]
            if not prior_bars.empty:
                benchmarks["PREVIOUS_DAY"] = SessionExtremes(
                    high=float(prior_bars["high"].max()),
                    low=float(prior_bars["low"].min()),
                    high_time=prior_bars["high"].idxmax(),
                    low_time=prior_bars["low"].idxmin(),
                    session_name="PREVIOUS_DAY"
                )

        return benchmarks

    def check_liquidation_sweep(
        self,
        cur_high: float,
        cur_low: float,
        bias: str,
        active_killzone: str,
        benchmarks: Dict[str, SessionExtremes],
        cur_time: Any
    ) -> LiquidationStatus:
        """
        Rule 3 Check: Determines whether price has swept the required session extreme.
        - For Bearish Shorts: Price MUST sweep a benchmark High.
        - For Bullish Longs: Price MUST sweep a benchmark Low.
        """
        if active_killzone == "LONDON_KILLZONE":
            # London shorts must sweep Asian High; London longs must sweep Asian Low
            if "ASIAN" in benchmarks:
                asian = benchmarks["ASIAN"]
                if bias == "BEARISH" and cur_high >= asian.high:
                    return LiquidationStatus(
                        swept=True,
                        sweep_type="HIGH_SWEPT",
                        benchmark_level=asian.high,
                        benchmark_name="Asian High (ASH)",
                        sweep_time=cur_time
                    )
                elif bias == "BULLISH" and cur_low <= asian.low:
                    return LiquidationStatus(
                        swept=True,
                        sweep_type="LOW_SWEPT",
                        benchmark_level=asian.low,
                        benchmark_name="Asian Low (ASL)",
                        sweep_time=cur_time
                    )

        elif active_killzone == "NEW_YORK_KILLZONE":
            # NY shorts must sweep London High or Premarket High or PDH
            # NY longs must sweep London Low or Premarket Low or PDL
            candidates = [k for k in ["PREMARKET", "LONDON", "PREVIOUS_DAY", "ASIAN"] if k in benchmarks]
            for c_key in candidates:
                b = benchmarks[c_key]
                if bias == "BEARISH" and cur_high >= b.high:
                    return LiquidationStatus(
                        swept=True,
                        sweep_type="HIGH_SWEPT",
                        benchmark_level=b.high,
                        benchmark_name=f"{b.session_name} High",
                        sweep_time=cur_time
                    )
                elif bias == "BULLISH" and cur_low <= b.low:
                    return LiquidationStatus(
                        swept=True,
                        sweep_type="LOW_SWEPT",
                        benchmark_level=b.low,
                        benchmark_name=f"{b.session_name} Low",
                        sweep_time=cur_time
                    )

        return LiquidationStatus(
            swept=False,
            sweep_type="NONE",
            benchmark_level=0.0,
            benchmark_name="NONE",
            sweep_time=None
        )
