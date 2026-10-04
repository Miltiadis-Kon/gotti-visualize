"""
SMC Intraday Bias Model Strategy Package
=========================================
Systematic implementation of Lewis Kelly's 5-Rule SMC Intraday Bias Model:
1. Rule 1: Directional Bias via The Box Method (15-Minute Macro Context)
2. Rule 2: Session Killzones (London 02:00-05:00 EST, NY 07:00-10:00/11:30 EST)
3. Rule 3: Liquidation (Asian/London/Premarket Extreme Sweeps)
4. Rule 4: Reversal Confirmation (1-Minute Micro-CHoCH Body Close)
5. Rule 5: Entry Model (5-Minute POI: Order Block, FVG, IFVG & Confluence)
6. Asymmetric Risk Architecture: Strict 1:5R Minimum Veto + Hybrid Partial Scaling (40% @ 3R + BE, 60% @ M15 External Target)
"""

from .bias_structure import (
    M15BoxStructureDetector,
    BiasDirection,
    ExternalSwingPoint,
    ExternalBoxRange,
    SwingPointType,
)
from .session_killzones import (
    SessionKillzoneManager,
    SessionExtremes,
    LiquidationStatus,
)
from .m1_choch import (
    M1CHoCHDetector,
    M1CHoCHConfirmation,
    MicroSwingPoint,
)
from .m5_poi import (
    M5POISelector,
    PointOfInterest,
    POIType,
)
from .bias_order_manager import (
    BiasOrderManager,
    BiasTradeRecord,
    BiasTradeTranche,
    OrderLifecycleStatus,
)
from .smc_bias_strategy import SMCIntradayBiasStrategy

# Standard aliases
SMCBiasModelStrategy = SMCIntradayBiasStrategy

__all__ = [
    "M15BoxStructureDetector",
    "BiasDirection",
    "ExternalSwingPoint",
    "ExternalBoxRange",
    "SwingPointType",
    "SessionKillzoneManager",
    "SessionExtremes",
    "LiquidationStatus",
    "M1CHoCHDetector",
    "M1CHoCHConfirmation",
    "MicroSwingPoint",
    "M5POISelector",
    "PointOfInterest",
    "POIType",
    "BiasOrderManager",
    "BiasTradeRecord",
    "BiasTradeTranche",
    "OrderLifecycleStatus",
    "SMCIntradayBiasStrategy",
    "SMCBiasModelStrategy",
]
