"""
ATR Risk Management Module
==========================
Dynamically optimizes Take Profit (TP) and Stop Loss (SL) based on
Average True Range (ATR) market volatility rather than arbitrary targets.

Framework:
    Long:
        SL = Entry - (ATR * M_SL)
        TP = Entry + (ATR * M_TP)  [where M_TP = M_SL * RR]
    Short:
        SL = Entry + (ATR * M_SL)
        TP = Entry - (ATR * M_TP)

Trading Styles:
    - Scalping:        SL 0.5x - 1.0x (default 1.0), RR 1:1 - 1:1.5 (default 1.5)
    - Day Trading:     SL 1.0x - 1.5x (default 1.5), RR 1:1.5 - 1:2 (default 2.0)
    - Swing Trading:   SL 2.0x - 3.0x (default 2.0), RR 1:2 - 1:3   (default 2.0)
    - Trend Following: SL 3.0x - 5.0x (default 3.0), Trailing Stop
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Trading Style Presets
# ─────────────────────────────────────────────────────────────────────────────

TRADING_STYLES: Dict[str, Dict[str, Any]] = {
    "scalping": {
        "name": "Scalping",
        "default_sl_multiplier": 1.0,
        "sl_multiplier": 1.0,
        "sl_range": (0.5, 1.0),
        "default_rr": 1.5,
        "risk_reward": 1.5,
        "rr_range": (1.0, 1.5),
        "trailing_stop": False,
        "description": "Requires tight stops for quick momentum bursts. High risk of being stopped out by standard noise.",
    },
    "day_trading": {
        "name": "Day Trading",
        "default_sl_multiplier": 1.5,
        "sl_multiplier": 1.5,
        "sl_range": (1.0, 1.5),
        "default_rr": 2.0,
        "risk_reward": 2.0,
        "rr_range": (1.5, 2.0),
        "trailing_stop": False,
        "description": "Clears the immediate noise of an average candle while keeping intra-day risk strictly contained.",
    },
    "swing_trading": {
        "name": "Swing Trading",
        "default_sl_multiplier": 2.0,
        "sl_multiplier": 2.0,
        "sl_range": (2.0, 3.0),
        "default_rr": 2.0,
        "risk_reward": 2.0,
        "rr_range": (2.0, 3.0),
        "trailing_stop": False,
        "description": "Allows the asset to undergo standard multi-day pullbacks without threatening the broader trend structure.",
    },
    "trend_following": {
        "name": "Trend Following",
        "default_sl_multiplier": 3.0,
        "sl_multiplier": 3.0,
        "sl_range": (3.0, 5.0),
        "default_rr": 3.0,
        "risk_reward": 3.0,
        "rr_range": (2.0, 4.0),
        "trailing_stop": True,
        "description": "Used for long-term trend riding (Chandelier Exit). Held until a massive reversal triggers the 3x-5x ATR stop.",
    },
}

# Alias standard abbreviations
TRADING_STYLES["day"] = TRADING_STYLES["day_trading"]
TRADING_STYLES["swing"] = TRADING_STYLES["swing_trading"]
TRADING_STYLES["trend"] = TRADING_STYLES["trend_following"]


# ─────────────────────────────────────────────────────────────────────────────
# Result Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ATRRiskLevels:
    """Holds calculated and volatility-optimized Stop Loss and Take Profit levels."""
    stop_loss: float
    take_profit: Optional[float]
    risk_distance: float
    reward_distance: Optional[float]
    atr: float
    sl_multiplier: float
    tp_multiplier: Optional[float]
    risk_reward_ratio: Optional[float]
    is_trailing: bool
    side: str
    entry_price: float
    custom_sl_raw: Optional[float] = None
    custom_tp_raw: Optional[float] = None
    adjusted_sl: bool = False
    adjusted_tp: bool = False

    def __iter__(self):
        """Allow tuple unpacking: stop_loss, take_profit = levels"""
        return iter((self.stop_loss, self.take_profit))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stop_loss": round(float(self.stop_loss), 4),
            "take_profit": round(float(self.take_profit), 4) if self.take_profit is not None else None,
            "risk_distance": round(float(self.risk_distance), 4),
            "reward_distance": round(float(self.reward_distance), 4) if self.reward_distance is not None else None,
            "atr": round(float(self.atr), 4),
            "sl_multiplier": self.sl_multiplier,
            "tp_multiplier": self.tp_multiplier,
            "risk_reward_ratio": self.risk_reward_ratio,
            "is_trailing": self.is_trailing,
            "side": self.side,
            "entry_price": round(float(self.entry_price), 4),
            "adjusted_sl": self.adjusted_sl,
            "adjusted_tp": self.adjusted_tp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ATR Indicator Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    """
    Calculate the latest Average True Range (ATR) from an OHLCV DataFrame.
    Gracefully falls back from pandas_ta to native pandas calculation if needed.
    """
    if df is None or df.empty or len(df) < 2:
        return 0.0

    # Normalize column names to lowercase
    col_map = {c: str(c).lower() for c in df.columns}
    sub_df = df.rename(columns=col_map)

    required = ["high", "low", "close"]
    if not all(k in sub_df.columns for k in required):
        return 0.0

    high = sub_df["high"].astype(float)
    low = sub_df["low"].astype(float)
    close = sub_df["close"].astype(float)

    # 1. Try pandas_ta if available
    try:
        import pandas_ta as ta
        atr_series = ta.atr(high=high, low=low, close=close, length=length)
        if atr_series is not None and not atr_series.empty:
            valid = atr_series.dropna()
            if not valid.empty:
                val = float(valid.iloc[-1])
                if not math.isnan(val) and val > 0:
                    return val
    except Exception:
        pass

    # 2. Native ATR Calculation: True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's Smoothing / RMA for ATR
    atr_series = true_range.ewm(alpha=1.0 / length, adjust=False).mean()
    valid = atr_series.dropna()
    if not valid.empty:
        val = float(valid.iloc[-1])
        if not math.isnan(val) and val > 0:
            return val

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core Stop Loss / Take Profit Optimization Function
# ─────────────────────────────────────────────────────────────────────────────

def optimize_sl_tp(
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    side: str = "buy",
    atr: Optional[float] = None,
    trading_style: str = "swing_trading",
    sl_multiplier: Optional[float] = None,
    tp_multiplier: Optional[float] = None,
    risk_reward_ratio: Optional[float] = None,
    trailing_stop: Optional[bool] = None,
    risk_reward: Optional[float] = None,
    **kwargs,
) -> ATRRiskLevels:
    """
    Optimizes Stop Loss (SL) and Take Profit (TP) using the ATR Multiplier Framework,
    adjusting dynamic volatility on top of any custom targets.

    Parameters:
        entry_price: The fill or execution price of the trade.
        stop_loss: Optional custom stop-loss calculated by the strategy.
        take_profit: Optional custom take-profit calculated by the strategy.
        side: Order side ('buy'/'long' or 'sell'/'short').
        atr: Current Average True Range value. If None or <= 0, defaults to 2% volatility estimate.
        trading_style: 'scalping', 'day_trading', 'swing_trading', or 'trend_following'.
        sl_multiplier: Custom multiplier override for SL (e.g. 2.0 for 2x ATR).
        tp_multiplier: Custom multiplier override for TP (e.g. 4.0 for 4x ATR).
        risk_reward_ratio: Custom Risk-to-Reward ratio override (e.g. 2.0 for 1:2).
        trailing_stop: Set True to enable trailing stop mechanics (standard for trend following).
        risk_reward: Alias for risk_reward_ratio.

    Returns:
        ATRRiskLevels dataclass which can be unpacked as (stop_loss, take_profit).
    """
    if risk_reward_ratio is None and risk_reward is not None:
        risk_reward_ratio = risk_reward

    if entry_price <= 0:
        raise ValueError(f"Entry price must be positive, got {entry_price}")

    # Normalize side
    side_lower = str(side).lower()
    is_long = side_lower in ("buy", "long")

    # Resolve style preset
    style_key = trading_style.lower() if trading_style else "swing_trading"
    style_cfg = TRADING_STYLES.get(style_key, TRADING_STYLES["swing_trading"])

    # Resolve multipliers and RR
    resolved_sl_mult = float(sl_multiplier) if sl_multiplier is not None else float(style_cfg["default_sl_multiplier"])
    sl_min_mult, sl_max_mult = style_cfg["sl_range"]
    resolved_is_trailing = bool(trailing_stop) if trailing_stop is not None else bool(style_cfg["trailing_stop"])

    # Fallback ATR if missing or non-positive (approx 2% of price)
    resolved_atr = float(atr) if (atr is not None and atr > 0) else (entry_price * 0.02)

    # 1. Base ATR Risk Distance
    risk_distance = resolved_atr * resolved_sl_mult

    # 2. Base ATR Reward Distance
    if tp_multiplier is not None:
        resolved_tp_mult = float(tp_multiplier)
        reward_distance = resolved_atr * resolved_tp_mult
        resolved_rr = reward_distance / risk_distance if risk_distance > 0 else 1.0
    else:
        resolved_rr = float(risk_reward_ratio) if risk_reward_ratio is not None else float(style_cfg["default_rr"])
        reward_distance = risk_distance * resolved_rr
        resolved_tp_mult = resolved_sl_mult * resolved_rr

    # 3. Base ATR Baseline Levels
    if is_long:
        base_sl = entry_price - risk_distance
        base_tp = entry_price + reward_distance
    else:
        base_sl = entry_price + risk_distance
        base_tp = entry_price - reward_distance

    # 4. Optimize Stop Loss on top of custom calculation
    final_sl = base_sl
    adjusted_sl = False

    if stop_loss is not None:
        custom_sl = float(stop_loss)
        # Calculate custom risk distance
        custom_risk_dist = (entry_price - custom_sl) if is_long else (custom_sl - entry_price)

        # Minimum breathing room distance to avoid premature noise stop-outs
        min_breathing_distance = resolved_atr * min(resolved_sl_mult, sl_min_mult)
        # Maximum risk threshold to prevent excessive capital loss
        max_risk_distance = resolved_atr * max(resolved_sl_mult, sl_max_mult)

        if custom_risk_dist <= 0:
            # Custom SL was invalid (wrong side of entry)
            final_sl = base_sl
            adjusted_sl = True
        elif custom_risk_dist < min_breathing_distance:
            # Too tight! High risk of whipsaw from market noise -> expand to breathing room
            final_sl = (entry_price - min_breathing_distance) if is_long else (entry_price + min_breathing_distance)
            adjusted_sl = True
        elif custom_risk_dist > max_risk_distance:
            # Too wide! Violates risk parameters -> contract to max envelope
            final_sl = (entry_price - max_risk_distance) if is_long else (entry_price + max_risk_distance)
            adjusted_sl = True
        else:
            # Custom stop loss is within healthy volatility range
            final_sl = custom_sl
            risk_distance = custom_risk_dist
    else:
        final_sl = base_sl

    # 5. Optimize Take Profit on top of custom calculation
    final_tp = base_tp if not resolved_is_trailing else None
    adjusted_tp = False

    if resolved_is_trailing:
        # Trend following uses trailing stop rather than a static take-profit
        final_tp = None
    elif take_profit is not None:
        custom_tp = float(take_profit)
        custom_reward_dist = (custom_tp - entry_price) if is_long else (entry_price - custom_tp)

        # Ensure reward distance matches at least the required Risk:Reward
        min_required_reward = risk_distance * min(resolved_rr, style_cfg["rr_range"][0])

        if custom_reward_dist <= 0:
            # Custom TP was invalid (wrong side of entry)
            final_tp = base_tp
            adjusted_tp = True
        elif custom_reward_dist < min_required_reward:
            # Arbitrary tight target under-rewards the trade relative to volatility
            final_tp = (entry_price + min_required_reward) if is_long else (entry_price - min_required_reward)
            reward_distance = min_required_reward
            adjusted_tp = True
        else:
            # Custom TP meets or exceeds volatility requirement
            final_tp = custom_tp
            reward_distance = custom_reward_dist
    else:
        final_tp = base_tp

    return ATRRiskLevels(
        stop_loss=final_sl,
        take_profit=final_tp,
        risk_distance=risk_distance,
        reward_distance=reward_distance if final_tp is not None else None,
        atr=resolved_atr,
        sl_multiplier=resolved_sl_mult,
        tp_multiplier=resolved_tp_mult if final_tp is not None else None,
        risk_reward_ratio=resolved_rr if final_tp is not None else None,
        is_trailing=resolved_is_trailing,
        side="buy" if is_long else "sell",
        entry_price=entry_price,
        custom_sl_raw=stop_loss,
        custom_tp_raw=take_profit,
        adjusted_sl=adjusted_sl,
        adjusted_tp=adjusted_tp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trailing Stop Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_trailing_stop(
    current_price: float = 0.0,
    arg2: Union[str, float, None] = None,
    arg3: Optional[float] = None,
    side: Optional[str] = None,
    risk_distance: Optional[float] = None,
    atr: Optional[float] = None,
    sl_multiplier: Optional[float] = None,
    current_stop: Optional[float] = None,
    highest_price: Optional[float] = None,
    lowest_price: Optional[float] = None,
    **kwargs,
) -> float:
    """
    Ratchets the stop loss following market price to lock in profits while
    maintaining the specified ATR risk breathing room.

    Parameters:
        current_price: Current market price (or highest_price for long, lowest_price for short).
        side: Position side ('buy'/'long' or 'sell'/'short').
        risk_distance: Exact dollar distance. If None, calculated as atr * sl_multiplier.
        atr: Current ATR value (used if risk_distance is None).
        sl_multiplier: Multiplier for ATR (default 2.0).
        current_stop: Existing stop price. Only ratchets forward (never loosens).
        highest_price: High water mark for long position trailing stop.
        lowest_price: Low water mark for short position trailing stop.
    """
    if highest_price is not None:
        current_price = highest_price
    elif lowest_price is not None:
        current_price = lowest_price

    # Disambiguate positional arg2 and arg3
    if isinstance(arg2, str):
        if side is None:
            side = arg2
        if isinstance(arg3, (int, float)) and risk_distance is None:
            risk_distance = float(arg3)
    elif isinstance(arg2, (int, float)):
        if atr is None:
            atr = float(arg2)
        if isinstance(arg3, (int, float)) and sl_multiplier is None:
            sl_multiplier = float(arg3)

    if side is None:
        side = kwargs.get("side", "buy")
    if sl_multiplier is None:
        sl_multiplier = 2.0

    is_long = str(side).lower() in ("buy", "long")

    dist = risk_distance
    if dist is None:
        if atr is not None and atr > 0:
            dist = atr * sl_multiplier
        else:
            dist = current_price * 0.02 * sl_multiplier

    if is_long:
        candidate_stop = current_price - dist
        if current_stop is None or current_stop <= 0:
            return candidate_stop
        return max(current_stop, candidate_stop)
    else:
        candidate_stop = current_price + dist
        if current_stop is None or current_stop <= 0:
            return candidate_stop
        return min(current_stop, candidate_stop)
