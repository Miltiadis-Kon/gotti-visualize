"""
SMC 3-Tier Multi-Tranche Position Manager (smc_tranche_manager.py)
==================================================================
Manages partial scaling and dynamic risk lifecycle:
  - Tranche 1 (40%): TP1 (Internal Liquidity) -> Closes 40%, moves remaining Stop Loss to Breakeven.
  - Tranche 2 (40%): TP2 (Primary Structural Target >= 3R) -> Closes 40%, initiates structural trailing stop.
  - Tranche 3 (20%): TP3 (External Liquidity Runner) -> Runs risk-free to PDH/PDL/EQH/EQL.
  - Invalidation Rule: Hard 1R stop loss maintained until TP1 is hit; no early breakeven moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import math


class TrancheStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TRAIL = "CLOSED_TRAIL"
    CANCELLED = "CANCELLED"


@dataclass
class PositionTranche:
    """Represents a partial slice of the overall trade."""
    tranche_id: str          # "TP1", "TP2", "TP3"
    target_pct: float        # 0.40, 0.40, 0.20
    quantity: int
    target_price: float
    target_name: str
    status: TrancheStatus = TrancheStatus.PENDING
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[Any] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None

    def close(self, price: float, time: Any, status: TrancheStatus, initial_risk_per_share: float) -> float:
        """Closes the tranche and calculates PnL and realized R-multiple."""
        self.exit_price = price
        self.exit_time = time
        self.status = status
        
        # PnL calculation
        if "BUY" in self.tranche_id or initial_risk_per_share > 0:
            price_delta = self.exit_price - self.entry_price
        else:
            price_delta = self.entry_price - self.exit_price

        self.pnl = price_delta * self.quantity
        if abs(initial_risk_per_share) > 1e-6:
            self.r_multiple = price_delta / abs(initial_risk_per_share)
        else:
            self.r_multiple = 0.0
        return self.pnl


@dataclass
class SMCTradeLifecycle:
    """Manages the complete multi-stage execution and risk lifecycle for one setup."""
    trade_id: int
    ticker: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    initial_stop_loss: float
    current_stop_loss: float
    total_quantity: int
    entry_time: Any
    
    # 3-Tier Targets
    tp1_price: float
    tp2_price: float
    tp3_price: float

    # Tranches
    tranches: Dict[str, PositionTranche] = field(default_factory=dict)
    
    # Lifecycle flags
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    is_breakeven: bool = False
    is_trailing: bool = False
    is_completed: bool = False
    completion_time: Optional[Any] = None
    total_realized_pnl: float = 0.0

    @property
    def remaining_quantity(self) -> int:
        return sum(t.quantity for t in self.tranches.values() if t.status == TrancheStatus.FILLED)

    @property
    def initial_risk_per_share(self) -> float:
        return abs(self.entry_price - self.initial_stop_loss)

    def check_exits(self, current_bar: pd.Series, timestamp: Any) -> List[Dict[str, Any]]:
        """
        Evaluates current 5-minute bar (high, low, close) to trigger partial TPs or Stop Losses.
        Returns a list of execution events occurred on this bar.
        """
        events: List[Dict[str, Any]] = []
        if self.is_completed:
            return events

        high = float(current_bar["high"])
        low = float(current_bar["low"])

        is_long = self.side.upper() == "BUY"

        # 1. Check Stop Loss FIRST
        sl_hit = (low <= self.current_stop_loss) if is_long else (high >= self.current_stop_loss)
        if sl_hit:
            exit_price = self.current_stop_loss
            # Close all remaining tranches at Stop Loss
            for tid, tranche in self.tranches.items():
                if tranche.status == TrancheStatus.FILLED:
                    status = TrancheStatus.CLOSED_SL if not self.is_breakeven else TrancheStatus.CLOSED_TRAIL
                    pnl = tranche.close(exit_price, timestamp, status, self.initial_risk_per_share)
                    self.total_realized_pnl += pnl
                    events.append({
                        "event": "STOP_LOSS_HIT",
                        "trade_id": self.trade_id,
                        "tranche": tid,
                        "quantity": tranche.quantity,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "r_multiple": tranche.r_multiple,
                        "is_breakeven": self.is_breakeven,
                    })
            self.is_completed = True
            self.completion_time = timestamp
            return events

        # 2. Check TP1 (40% - Internal Liquidity)
        t1 = self.tranches.get("TP1")
        if t1 and t1.status == TrancheStatus.FILLED and not self.tp1_hit:
            t1_reached = (high >= self.tp1_price) if is_long else (low <= self.tp1_price)
            if t1_reached:
                self.tp1_hit = True
                pnl = t1.close(self.tp1_price, timestamp, TrancheStatus.CLOSED_TP, self.initial_risk_per_share)
                self.total_realized_pnl += pnl
                
                # RULE: Move Stop Loss to Breakeven for remaining 60%
                self.current_stop_loss = self.entry_price
                self.is_breakeven = True

                events.append({
                    "event": "TP1_SCALED_OUT",
                    "trade_id": self.trade_id,
                    "tranche": "TP1",
                    "quantity": t1.quantity,
                    "exit_price": self.tp1_price,
                    "pnl": pnl,
                    "r_multiple": t1.r_multiple,
                    "action": "SL_MOVED_TO_BREAKEVEN",
                    "new_sl": self.current_stop_loss,
                })

        # 3. Check TP2 (40% - Primary Structural Target >= 3R)
        t2 = self.tranches.get("TP2")
        if t2 and t2.status == TrancheStatus.FILLED and not self.tp2_hit:
            t2_reached = (high >= self.tp2_price) if is_long else (low <= self.tp2_price)
            if t2_reached:
                self.tp2_hit = True
                pnl = t2.close(self.tp2_price, timestamp, TrancheStatus.CLOSED_TP, self.initial_risk_per_share)
                self.total_realized_pnl += pnl

                # RULE: Activate trailing stop behind structural swings for runner
                self.is_trailing = True

                events.append({
                    "event": "TP2_SCALED_OUT",
                    "trade_id": self.trade_id,
                    "tranche": "TP2",
                    "quantity": t2.quantity,
                    "exit_price": self.tp2_price,
                    "pnl": pnl,
                    "r_multiple": t2.r_multiple,
                    "action": "TRAILING_ACTIVATED_FOR_RUNNER",
                })

        # 4. Check TP3 (20% - External Liquidity Runner)
        t3 = self.tranches.get("TP3")
        if t3 and t3.status == TrancheStatus.FILLED and not self.tp3_hit:
            t3_reached = (high >= self.tp3_price) if is_long else (low <= self.tp3_price)
            if t3_reached:
                self.tp3_hit = True
                pnl = t3.close(self.tp3_price, timestamp, TrancheStatus.CLOSED_TP, self.initial_risk_per_share)
                self.total_realized_pnl += pnl

                events.append({
                    "event": "TP3_RUNNER_HIT",
                    "trade_id": self.trade_id,
                    "tranche": "TP3",
                    "quantity": t3.quantity,
                    "exit_price": self.tp3_price,
                    "pnl": pnl,
                    "r_multiple": t3.r_multiple,
                    "action": "TRADE_COMPLETED_MAX_REWARD",
                })

        # Check if all tranches are closed
        if all(t.status in (TrancheStatus.CLOSED_TP, TrancheStatus.CLOSED_SL, TrancheStatus.CLOSED_TRAIL) for t in self.tranches.values()):
            self.is_completed = True
            self.completion_time = timestamp

        return events

    def update_trailing_stop(self, new_structural_level: float) -> bool:
        """
        Updates the trailing stop for Tranche 3 runner behind new structural swing pivots.
        Only moves stop loss in favor of the trade (ratchet).
        """
        if not self.is_trailing or self.is_completed:
            return False

        is_long = self.side.upper() == "BUY"
        if is_long:
            if new_structural_level > self.current_stop_loss:
                self.current_stop_loss = new_structural_level
                return True
        else:
            if new_structural_level < self.current_stop_loss:
                self.current_stop_loss = new_structural_level
                return True
        return False


class SMCTrancheManager:
    """
    Orchestrates creation, allocation, and tracking of 3-tier partial positions.
    Default distribution: 40% TP1, 40% TP2, 20% TP3.
    """

    def __init__(
        self,
        tp1_pct: float = 0.40,
        tp2_pct: float = 0.40,
        tp3_pct: float = 0.20,
    ):
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.tp3_pct = tp3_pct
        self.trades: Dict[int, SMCTradeLifecycle] = {}
        self._next_id = 1

    def create_trade(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        tp1_price: float,
        tp2_price: float,
        tp3_price: float,
        total_quantity: int,
        entry_time: Any,
    ) -> SMCTradeLifecycle:
        """Splits total position into 3 discrete tranches and initializes trade lifecycle."""
        q1 = max(1, int(math.floor(total_quantity * self.tp1_pct)))
        q2 = max(1, int(math.floor(total_quantity * self.tp2_pct)))
        q3 = max(1, total_quantity - q1 - q2)

        trade = SMCTradeLifecycle(
            trade_id=self._next_id,
            ticker=ticker,
            side=side.upper(),
            entry_price=entry_price,
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            total_quantity=total_quantity,
            entry_time=entry_time,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
        )

        trade.tranches = {
            "TP1": PositionTranche(
                tranche_id="TP1",
                target_pct=self.tp1_pct,
                quantity=q1,
                target_price=tp1_price,
                target_name="Internal Liquidity (TP1)",
                status=TrancheStatus.FILLED,
                entry_price=entry_price,
            ),
            "TP2": PositionTranche(
                tranche_id="TP2",
                target_pct=self.tp2_pct,
                quantity=q2,
                target_price=tp2_price,
                target_name="Primary Structural Target (TP2)",
                status=TrancheStatus.FILLED,
                entry_price=entry_price,
            ),
            "TP3": PositionTranche(
                tranche_id="TP3",
                target_pct=self.tp3_pct,
                quantity=q3,
                target_price=tp3_price,
                target_name="External Liquidity Runner (TP3)",
                status=TrancheStatus.FILLED,
                entry_price=entry_price,
            ),
        }

        self.trades[self._next_id] = trade
        self._next_id += 1
        return trade
