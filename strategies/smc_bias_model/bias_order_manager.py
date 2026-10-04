"""
SMC Bias Model: Order Execution & Asymmetric Risk Manager (bias_order_manager.py)
==================================================================================
Section 4 & User Constraints:
- Risk Architecture:
  * Stop Loss: 1-2 ticks/cents beyond the microscopic M1/M5 structural invalidation extreme.
  * Entry: Proximate boundary of the chosen M5 POI (OB, FVG, IFVG).
  * Take Profit Target: 15-Minute external swing low (shorts) / high (longs).
- Strict Asymmetric 1:5R Veto Rule:
  * Setup MUST yield >= 5.0R between Entry and M15 External Target.
  * If R:R < 5.0, setup is strictly VETOED immediately.
- Hybrid Partial Scaling Model (User confirmed):
  * TP1 (40%): Scaled out at 3.0R -> Stop Loss moved to Entry Price (Breakeven).
  * TP2 / Runner (60%): Rides to the 15-minute external swing target (>= 5.0R).
  * Stopped out before TP1: 100% loss at 1.0R.
  * Stopped out after TP1: Remaining 60% stopped out at breakeven ($0 PnL, locking in net +1.2R gain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import pandas as pd


class OrderLifecycleStatus(Enum):
    PENDING_LIMIT = "PENDING_LIMIT"
    IN_POSITION = "IN_POSITION"
    TP1_SCALED = "TP1_SCALED"
    COMPLETED_WIN = "COMPLETED_WIN"
    COMPLETED_LOSS = "COMPLETED_LOSS"
    COMPLETED_BREAKEVEN = "COMPLETED_BREAKEVEN"
    CANCELLED = "CANCELLED"


@dataclass
class BiasTradeTranche:
    tranche_id: str  # "TP1_40" or "RUNNER_60"
    target_price: float
    target_r: float
    quantity: int
    exit_price: Optional[float] = None
    exit_time: Optional[Any] = None
    pnl: float = 0.0
    status: str = "PENDING"


@dataclass
class BiasTradeRecord:
    trade_id: int
    ticker: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    initial_stop_loss: float
    current_stop_loss: float
    tp1_price: float
    tp2_target_price: float
    total_quantity: int
    entry_time: Any
    risk_per_share: float
    rr_ratio: float
    status: OrderLifecycleStatus = OrderLifecycleStatus.IN_POSITION
    tp1_hit: bool = False
    tp2_hit: bool = False
    stopped_out: bool = False
    total_realized_pnl: float = 0.0
    completion_time: Optional[Any] = None
    tranches: Dict[str, BiasTradeTranche] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status in (
            OrderLifecycleStatus.COMPLETED_WIN,
            OrderLifecycleStatus.COMPLETED_LOSS,
            OrderLifecycleStatus.COMPLETED_BREAKEVEN,
            OrderLifecycleStatus.CANCELLED,
        )

    def check_exits(self, cur_bar: pd.Series, cur_time: Any) -> List[Dict[str, Any]]:
        """
        Evaluates bar price action (high/low) against current stop loss, TP1 (3.0R), and TP2 (>=5.0R).
        """
        if self.is_completed:
            return []

        events = []
        cur_high = float(cur_bar["high"])
        cur_low = float(cur_bar["low"])

        # 1. Stop Loss Check
        sl_hit = (cur_low <= self.current_stop_loss) if self.side == "BUY" else (cur_high >= self.current_stop_loss)
        if sl_hit:
            exit_p = self.current_stop_loss
            remaining_pnl = 0.0

            for tid, tranche in self.tranches.items():
                if tranche.status == "PENDING":
                    tranche.status = "STOPPED_OUT"
                    tranche.exit_price = exit_p
                    tranche.exit_time = cur_time
                    pnl_t = (exit_p - self.entry_price) * tranche.quantity if self.side == "BUY" else (self.entry_price - exit_p) * tranche.quantity
                    tranche.pnl = pnl_t
                    remaining_pnl += pnl_t

            self.total_realized_pnl += remaining_pnl
            self.completion_time = cur_time
            self.stopped_out = True

            if self.tp1_hit:
                self.status = OrderLifecycleStatus.COMPLETED_BREAKEVEN
            else:
                self.status = OrderLifecycleStatus.COMPLETED_LOSS

            events.append({
                "event": "STOP_LOSS_HIT",
                "exit_price": exit_p,
                "time": cur_time,
                "net_pnl": self.total_realized_pnl
            })
            return events

        # 2. TP1 Check (40% at 3.0R)
        if not self.tp1_hit:
            tp1_hit = (cur_high >= self.tp1_price) if self.side == "BUY" else (cur_low <= self.tp1_price)
            if tp1_hit:
                self.tp1_hit = True
                t1 = self.tranches.get("TP1_40")
                if t1 and t1.status == "PENDING":
                    t1.status = "FILLED"
                    t1.exit_price = self.tp1_price
                    t1.exit_time = cur_time
                    pnl_t1 = (self.tp1_price - self.entry_price) * t1.quantity if self.side == "BUY" else (self.entry_price - self.tp1_price) * t1.quantity
                    t1.pnl = pnl_t1
                    self.total_realized_pnl += pnl_t1

                # MOVE STOP LOSS TO BREAKEVEN (Entry Price)
                self.current_stop_loss = self.entry_price
                self.status = OrderLifecycleStatus.TP1_SCALED
                events.append({
                    "event": "TP1_SCALED_OUT",
                    "exit_price": self.tp1_price,
                    "time": cur_time,
                    "pnl": pnl_t1,
                    "new_sl": self.current_stop_loss
                })

        # 3. TP2 Check (60% Runner at M15 External Target >= 5.0R)
        if self.tp1_hit and not self.tp2_hit:
            tp2_hit = (cur_high >= self.tp2_target_price) if self.side == "BUY" else (cur_low <= self.tp2_target_price)
            if tp2_hit:
                self.tp2_hit = True
                t2 = self.tranches.get("RUNNER_60")
                if t2 and t2.status == "PENDING":
                    t2.status = "FILLED"
                    t2.exit_price = self.tp2_target_price
                    t2.exit_time = cur_time
                    pnl_t2 = (self.tp2_target_price - self.entry_price) * t2.quantity if self.side == "BUY" else (self.entry_price - self.tp2_target_price) * t2.quantity
                    t2.pnl = pnl_t2
                    self.total_realized_pnl += pnl_t2

                self.status = OrderLifecycleStatus.COMPLETED_WIN
                self.completion_time = cur_time
                events.append({
                    "event": "TP2_RUNNER_HIT",
                    "exit_price": self.tp2_target_price,
                    "time": cur_time,
                    "pnl": pnl_t2,
                    "total_realized_pnl": self.total_realized_pnl
                })

        return events


class BiasOrderManager:
    """
    Manages order creation, validates the Asymmetric 1:5R Veto Rule,
    sizes positions, and orchestrates the Hybrid Partial Scaling lifecycle.
    """

    def __init__(
        self,
        min_rr: float = 5.0,
        risk_pct: float = 0.01,
        tp1_pct: float = 0.40,
        tp2_pct: float = 0.60,
        tick_size: float = 0.01
    ):
        self.min_rr = min_rr
        self.risk_pct = risk_pct
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.tick_size = tick_size
        self._next_id: int = 1

    def evaluate_5r_veto(
        self,
        entry_price: float,
        stop_loss: float,
        m15_target: float,
        side: str
    ) -> Tuple[bool, float, str]:
        """
        Asymmetric R:R Gate: Enforces that the macro M15 external swing target
        yields at least 1:5.0R relative to the microscopic structural stop loss,
        with strict directional integrity.
        """
        if side == "BUY":
            if stop_loss >= entry_price:
                return False, 0.0, "VETO: Stop loss is above or equal to entry for a BUY."
            if m15_target <= entry_price:
                return False, 0.0, "VETO: Target is below or equal to entry for a BUY."
            risk = entry_price - stop_loss
            reward = m15_target - entry_price
        elif side == "SELL":
            if stop_loss <= entry_price:
                return False, 0.0, "VETO: Stop loss is below or equal to entry for a SELL."
            if m15_target >= entry_price:
                return False, 0.0, "VETO: Target is above or equal to entry for a SELL."
            risk = stop_loss - entry_price
            reward = entry_price - m15_target
        else:
            return False, 0.0, f"VETO: Invalid side '{side}'."

        if risk <= 0:
            return False, 0.0, "VETO: Risk per share must be positive."

        rr = reward / risk
        if rr < self.min_rr:
            return False, rr, f"VETO: Setup R:R ({rr:.2f}R) does not satisfy the strict 1:{self.min_rr:.1f}R minimum."

        return True, rr, f"APPROVED: Asymmetric R:R = {rr:.2f}R (>= {self.min_rr:.1f}R)."

    def create_trade(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        structural_extreme_price: float,
        m15_target_price: float,
        equity: float,
        entry_time: Any
    ) -> Optional[BiasTradeRecord]:
        """
        Creates a new trade with Hybrid Partial Scaling:
        - Structural Stop Loss: 1 tick beyond extreme
        - TP1 (40%): At 3.0R
        - TP2 (60%): At M15 External Target (>= 5.0R)
        """
        # Stop loss with 1 tick buffer
        sl_price = (structural_extreme_price - self.tick_size) if side == "BUY" else (structural_extreme_price + self.tick_size)
        risk_per_share = abs(entry_price - sl_price)

        approved, rr, msg = self.evaluate_5r_veto(entry_price, sl_price, m15_target_price, side)
        if not approved:
            return None

        # Position Sizing
        total_risk_budget = equity * self.risk_pct
        total_qty = max(2, int(math.floor(total_risk_budget / risk_per_share)))

        qty_tp1 = max(1, int(math.floor(total_qty * self.tp1_pct)))
        qty_tp2 = total_qty - qty_tp1

        # TP1 level at 3.0R
        tp1_price = (entry_price + 3.0 * risk_per_share) if side == "BUY" else (entry_price - 3.0 * risk_per_share)

        tranches = {
            "TP1_40": BiasTradeTranche(
                tranche_id="TP1_40",
                target_price=tp1_price,
                target_r=3.0,
                quantity=qty_tp1,
                status="PENDING"
            ),
            "RUNNER_60": BiasTradeTranche(
                tranche_id="RUNNER_60",
                target_price=m15_target_price,
                target_r=rr,
                quantity=qty_tp2,
                status="PENDING"
            )
        }

        trade = BiasTradeRecord(
            trade_id=self._next_id,
            ticker=ticker,
            side=side,
            entry_price=entry_price,
            initial_stop_loss=sl_price,
            current_stop_loss=sl_price,
            tp1_price=tp1_price,
            tp2_target_price=m15_target_price,
            total_quantity=total_qty,
            entry_time=entry_time,
            risk_per_share=risk_per_share,
            rr_ratio=rr,
            status=OrderLifecycleStatus.IN_POSITION,
            tranches=tranches
        )
        self._next_id += 1
        return trade
