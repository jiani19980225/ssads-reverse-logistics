"""Recovery value, processing rate, inspection savings, and throughput."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetrics:
    TRV: float    # Total Recovery Value (net USD)
    RPR: float    # Recovery Processing Rate (share of arrivals processed)
    ICS: float    # Inspection Cost Savings vs inspect-all (USD)
    TPR: float    # Throughput Rate (assets/hour)
    n_assets: int
    frac_inspected: float = 0.0
    inspection_skip: int = 0
    inspection_quick: int = 0
    inspection_full: int = 0

def compute_metrics(results: list[dict], full_inspection_cost: float) -> RunMetrics:
    """Compute metrics from a list of per-asset result dicts.

    Each result dict must have:
        realized_value: float (revenue - processing cost)
        inspection_cost: float
        disposition: str (for example "component_recovery", "refurbish", "scrap")
        time_min: float
    """
    if isinstance(full_inspection_cost, bool) or not isinstance(full_inspection_cost, (int, float)):
        raise TypeError("full_inspection_cost must be numeric")
    if not math.isfinite(float(full_inspection_cost)) or full_inspection_cost < 0:
        raise ValueError("full_inspection_cost must be finite and nonnegative")
    required = {"realized_value", "inspection_cost", "inspection_level", "disposition", "time_min"}
    for idx, result in enumerate(results):
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(f"Result {idx} is missing fields: {', '.join(missing)}")
        for field in ("realized_value", "inspection_cost", "time_min"):
            value = result[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Result {idx} {field} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"Result {idx} {field} must be finite")
        if result["inspection_cost"] < 0 or result["time_min"] < 0:
            raise ValueError(f"Result {idx} costs and time must be nonnegative")
        if result["inspection_level"] not in (0, 1, 2):
            raise ValueError(f"Result {idx} has an invalid inspection level")
        if result["disposition"] not in {
            "component_recovery", "partial_recovery", "refurbish", "scrap",
        }:
            raise ValueError(f"Result {idx} has an invalid disposition")

    n = len(results)
    if n == 0:
        return RunMetrics(TRV=0, RPR=0, ICS=0, TPR=0, n_assets=0, frac_inspected=0.0)

    trv = sum(r["realized_value"] - r["inspection_cost"] for r in results)
    n_processed = sum(1 for r in results if r["disposition"] != "scrap")
    rpr = n_processed / n
    total_insp = sum(r["inspection_cost"] for r in results)
    ics = full_inspection_cost * n - total_insp
    total_time_hr = sum(r["time_min"] for r in results) / 60.0
    if total_time_hr <= 0:
        raise ValueError("Non-empty results must have positive total processing time")
    tpr = n / total_time_hr
    levels = [int(r.get("inspection_level", 0)) for r in results]
    frac_inspected = sum(level > 0 for level in levels) / n

    return RunMetrics(
        TRV=trv,
        RPR=rpr,
        ICS=ics,
        TPR=tpr,
        n_assets=n,
        frac_inspected=frac_inspected,
        inspection_skip=levels.count(0),
        inspection_quick=levels.count(1),
        inspection_full=levels.count(2),
    )
