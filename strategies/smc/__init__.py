"""
Smart Money Concepts (SMC) Algorithmic Package
===============================================
Institutional order flow, structure, liquidity, and multi-timeframe execution.
"""

from .smc_structure import (
    FractalPivotDetector,
    MarketStructure,
    SwingPoint,
    StructureType,
    detect_structure_shifts,
)
from .smc_fvg import (
    FairValueGap,
    FVGType,
    detect_fvgs,
    filter_fvgs_by_equilibrium,
    update_fvg_mitigation,
)
from .smc_liquidity import (
    LiquidityPool,
    LiquidityType,
    find_liquidity_pools,
    evaluate_3r_veto,
)
from .smc_tranche_manager import SMCTrancheManager, PositionTranche
from .smc_strategy import SmartMoneyConceptsStrategy

# Standard aliases
SMCStrategy = SmartMoneyConceptsStrategy

__all__ = [
    "FractalPivotDetector",
    "MarketStructure",
    "SwingPoint",
    "StructureType",
    "detect_structure_shifts",
    "FairValueGap",
    "FVGType",
    "detect_fvgs",
    "filter_fvgs_by_equilibrium",
    "update_fvg_mitigation",
    "LiquidityPool",
    "LiquidityType",
    "find_liquidity_pools",
    "evaluate_3r_veto",
    "SMCTrancheManager",
    "PositionTranche",
    "SmartMoneyConceptsStrategy",
    "SMCStrategy",
]
