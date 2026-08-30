"""Adaptive Inspection Policy: three-tier threshold rule.

From Section III-C:
    sigma >= tau_h: skip inspection (level 0)
    tau_l <= sigma < tau_h: quick L1 test (level 1)
    sigma < tau_l: full L2 test (level 2)
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionDecision:
    level: int          # 0=skip, 1=quick, 2=full
    cost: float         # USD
    time_min: float     # minutes


def adaptive_inspection(
    sigma: float,
    tau_h: float,
    tau_l: float,
    costs: dict,
) -> InspectionDecision:
    """Determine inspection depth from a signal-quality score.

    Args:
        sigma: extractor signal quality in [0, 1].
        tau_h: high threshold (skip if above).
        tau_l: low threshold (full inspect if below).
        costs: dict with keys 'skip', 'l1', 'l2', each having 'cost' and 'time_min'.
    """
    for name, value in (("sigma", sigma), ("tau_h", tau_h), ("tau_l", tau_l)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if tau_l > tau_h:
        raise ValueError("tau_l cannot exceed tau_h")
    if sigma >= tau_h:
        c = costs["skip"]
        return InspectionDecision(level=0, cost=c["cost"], time_min=c["time_min"])
    elif sigma >= tau_l:
        c = costs["l1"]
        return InspectionDecision(level=1, cost=c["cost"], time_min=c["time_min"])
    else:
        c = costs["l2"]
        return InspectionDecision(level=2, cost=c["cost"], time_min=c["time_min"])


def inspect_all(costs: dict) -> InspectionDecision:
    """Always perform full inspection."""
    c = costs["l2"]
    return InspectionDecision(level=2, cost=c["cost"], time_min=c["time_min"])


def skip_inspection(costs: dict) -> InspectionDecision:
    """Use no physical inspection."""
    c = costs["skip"]
    return InspectionDecision(level=0, cost=c["cost"], time_min=c["time_min"])


def inspection_at_level(level: int, costs: dict) -> InspectionDecision:
    """Construct a decision for an explicit inspection level."""
    if level == 0:
        return skip_inspection(costs)
    if level == 1:
        c = costs["l1"]
        return InspectionDecision(level=1, cost=c["cost"], time_min=c["time_min"])
    if level == 2:
        return inspect_all(costs)
    raise ValueError(f"Unknown inspection level: {level}")
