"""Capacity-constrained recovery allocation.

Every first-stage inspection is charged before allocation. The allocator then
compares recovery actions with the default scrap disposition and uses the labor
remaining after both inspection and default scrap handling. Realized component
yields are stochastic, but expected action margins are analytical.
"""

from __future__ import annotations

import math

import numpy as np

from .beta_model import ground_truth_params, sample_yield


class CapacityInfeasibleError(ValueError):
    """The first-stage plan and default dispositions exceed available labor."""


def _finite(value, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and out > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return out


def _component_expected_value(
    components: dict[str, int],
    prices: dict[str, float],
    base_yields: dict[str, float],
    phi: float,
) -> float:
    phi = _finite(phi, "phi", minimum=0.0, maximum=1.0)
    if phi == 0:
        raise ValueError("phi must be greater than zero")
    value = 0.0
    for name, count in components.items():
        if name not in prices or name not in base_yields:
            raise KeyError(f"Missing price or base yield for component {name!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Component count for {name!r} must be a positive integer")
        price = _finite(prices[name], f"prices.{name}", minimum=0.0)
        base_yield = _finite(base_yields[name], f"base_yields.{name}",
                             minimum=0.0, maximum=1.0)
        value += count * price * base_yield * phi
    return float(value)


def compute_expected_value(
    components: dict[str, int],
    prices: dict[str, float],
    base_yields: dict[str, float],
    phi: float,
    refurb_value: float = 0.0,
    processing_costs: dict[str, dict] | None = None,
) -> tuple[float, str]:
    """Return the expected gross value and best recovery action.

    When action costs are supplied, "best" means highest expected net value.
    The optional cost-free form remains useful for isolated value calculations.
    """
    component_value = _component_expected_value(components, prices, base_yields, phi)
    refurb_value = _finite(refurb_value, "refurb_value", minimum=0.0)
    candidates = [(component_value, "l2")]
    if refurb_value > 0:
        candidates.append((refurb_value * phi, "l0_refurb"))

    if processing_costs is None:
        gross, action = max(candidates, key=lambda item: item[0])
        return float(gross), action

    available = []
    for gross, action in candidates:
        if action not in processing_costs:
            if action == "l2":
                raise KeyError("processing_costs.l2 is required")
            continue
        cost = _finite(processing_costs[action]["cost"],
                       f"processing_costs.{action}.cost", minimum=0.0)
        available.append((gross - cost, gross, action))
    if not available:
        raise ValueError("No configured recovery action is available")
    _, gross, action = max(available, key=lambda item: item[0])
    return float(gross), action


def realize_value(
    components: dict[str, int],
    prices: dict[str, float],
    true_yield_factor: float,
    base_yields: dict[str, float],
    rng: np.random.Generator,
    concentration: float = 20.0,
) -> float:
    """Draw realized component value from the ground-truth Beta model."""
    true_yield_factor = _finite(
        true_yield_factor, "true_yield_factor", minimum=0.0, maximum=1.0
    )
    comp_names = list(components)
    if not comp_names:
        return 0.0
    for name in comp_names:
        if name not in prices or name not in base_yields:
            raise KeyError(f"Missing price or base yield for component {name!r}")
    counts = np.array([components[name] for name in comp_names], dtype=float)
    prices_arr = np.array([prices[name] for name in comp_names], dtype=float)
    base_arr = np.array([base_yields[name] for name in comp_names], dtype=float)
    if np.any(counts <= 0) or not np.all(np.isfinite(counts)):
        raise ValueError("Component counts must be finite and positive")
    if np.any(prices_arr < 0) or not np.all(np.isfinite(prices_arr)):
        raise ValueError("Component prices must be finite and nonnegative")
    if np.any((base_arr < 0) | (base_arr > 1)) or not np.all(np.isfinite(base_arr)):
        raise ValueError("Base yields must be finite and in [0, 1]")

    # A configured zero baseline yield is a structural zero, not a small positive
    # Beta mean introduced only to keep the distribution numerically well-defined.
    mask = (prices_arr > 0) & (base_arr > 0)
    if not np.any(mask):
        return 0.0
    params = ground_truth_params(
        np.minimum(true_yield_factor * base_arr[mask], 0.99),
        concentration=concentration,
    )
    actual = np.asarray(sample_yield(params, rng, n=1), dtype=float)
    return float(actual @ (prices_arr[mask] * counts[mask]))


def prepare_realized_outcomes(
    assets: list[dict],
    prices: dict[str, float],
    base_yields: dict[str, float],
    rng: np.random.Generator,
    concentration: float = 20.0,
) -> None:
    """Pre-draw per-asset component outcomes for paired common-random-number runs."""
    for asset in assets:
        asset["realized_component_value"] = realize_value(
            asset["components"], prices, asset["true_yield_factor"], base_yields, rng,
            concentration=concentration,
        )


def score_assets(
    assets: list[dict],
    prices: dict[str, float],
    base_yields: dict[str, float],
    processing_costs: dict[str, dict],
) -> list[dict]:
    """Attach the best recovery action and its incremental scrap-relative margin."""
    scrap = processing_costs["scrap"]
    scrap_cost = _finite(scrap["cost"], "processing_costs.scrap.cost", minimum=0.0)
    scrap_time = _finite(scrap["time_min"], "processing_costs.scrap.time_min", minimum=0.0)
    refurb_values = processing_costs.get("refurb_values", {})
    for asset in assets:
        refurb_value = float(refurb_values.get(asset.get("asset_type", ""), 0.0))
        component_expected = _component_expected_value(
            asset["components"], prices, base_yields, asset["phi"]
        )
        expected_value, best_level = compute_expected_value(
            asset["components"], prices, base_yields, asset["phi"],
            refurb_value=refurb_value, processing_costs=processing_costs,
        )
        proc = processing_costs[best_level]
        proc_cost = _finite(proc["cost"], f"processing_costs.{best_level}.cost", minimum=0.0)
        proc_time = _finite(
            proc["time_min"], f"processing_costs.{best_level}.time_min", minimum=0.0
        )
        capacity_time = proc_time - scrap_time
        if capacity_time < -1e-9:
            raise ValueError(f"Recovery action {best_level} cannot use less time than scrap")
        margin = expected_value - proc_cost + scrap_cost
        asset.update({
            "expected_component_value": component_expected,
            "expected_value": expected_value,
            "best_level": best_level,
            "proc_cost": proc_cost,
            "proc_time": proc_time,
            "scrap_cost": scrap_cost,
            "scrap_time": scrap_time,
            "capacity_time": max(0.0, capacity_time),
            "processing_margin": margin,
            "value_density": margin / capacity_time if capacity_time > 0 else float("inf"),
        })
    return assets


def inspection_minutes(assets: list[dict]) -> float:
    return float(sum(asset["inspection"].time_min for asset in assets))


def remaining_processing_capacity(
    assets: list[dict],
    capacity_minutes: float,
    processing_costs: dict[str, dict] | None = None,
) -> float:
    """Capacity left for replacing default scrap dispositions with recovery."""
    capacity_minutes = _finite(capacity_minutes, "capacity_minutes", minimum=0.0)
    if processing_costs is not None:
        scrap_time = _finite(
            processing_costs["scrap"]["time_min"],
            "processing_costs.scrap.time_min", minimum=0.0,
        )
        for asset in assets:
            asset.setdefault("scrap_time", scrap_time)
            asset.setdefault("scrap_cost", float(processing_costs["scrap"]["cost"]))
    try:
        default_scrap_minutes = sum(float(asset["scrap_time"]) for asset in assets)
    except KeyError as exc:
        raise ValueError("Assets must be scored or processing_costs must be supplied") from exc
    remaining = capacity_minutes - inspection_minutes(assets) - default_scrap_minutes
    if remaining < -1e-9:
        raise CapacityInfeasibleError(
            "Inspection policy and default scrap handling exceed shared labor capacity"
        )
    return max(0.0, float(remaining))


def _scrap_result(asset: dict) -> dict:
    return {
        "realized_value": -float(asset["scrap_cost"]),
        "inspection_cost": asset["inspection"].cost,
        "inspection_level": asset["inspection"].level,
        "disposition": "scrap",
        "time_min": asset["inspection"].time_min + float(asset["scrap_time"]),
    }


def rework_adjustment(asset: dict, processing_costs: dict[str, dict]) -> tuple[float, float]:
    """Return realized downstream rework cost and time for a risky skipped asset."""
    rework = processing_costs.get("rework")
    if not rework:
        return 0.0, 0.0
    trigger = float(rework.get("trigger_yield_below", 0.30))
    if asset["inspection"].level == 0 and asset["true_yield_factor"] < trigger:
        return float(rework["cost"]), float(rework.get("time_min", 0.0))
    return 0.0, 0.0


def apply_rework_cost(asset: dict, realized: float, processing_costs: dict[str, dict]) -> float:
    """Backward-compatible value-only wrapper around :func:`rework_adjustment`."""
    cost, _ = rework_adjustment(asset, processing_costs)
    return float(realized) - cost


def _capacity_time(asset: dict) -> float:
    return float(asset.get("capacity_time", asset["proc_time"]))


def greedy_expected_objective(
    assets: list[dict],
    processing_capacity_minutes: float,
    disable_ranking: bool = False,
) -> tuple[float, set[int]]:
    """Return heuristic expected incremental margin and selected identities."""
    candidates = [asset for asset in assets if asset["processing_margin"] > 0]
    if not disable_ranking:
        candidates = sorted(
            candidates,
            key=lambda asset: (-asset["value_density"], -asset["processing_margin"]),
        )

    used = 0.0
    objective = 0.0
    selected: set[int] = set()
    for asset in candidates:
        action_time = _capacity_time(asset)
        if used + action_time <= processing_capacity_minutes + 1e-9:
            used += action_time
            objective += asset["processing_margin"]
            selected.add(id(asset))
    return float(objective), selected


def exact_expected_objective(
    assets: list[dict], processing_capacity_minutes: float
) -> float:
    """Solve the fixed-action 0-1 allocation problem exactly with SciPy MILP."""
    processing_capacity_minutes = _finite(
        processing_capacity_minutes, "processing_capacity_minutes", minimum=0.0
    )
    candidates = [
        asset for asset in assets
        if asset["processing_margin"] > 0
        and _capacity_time(asset) <= processing_capacity_minutes + 1e-9
    ]
    if not candidates:
        return 0.0

    values = np.array([asset["processing_margin"] for asset in candidates], dtype=float)
    times = np.array([_capacity_time(asset) for asset in candidates], dtype=float)
    if times.sum() <= processing_capacity_minutes + 1e-9:
        return float(values.sum())

    unique_times = np.unique(times)
    if len(unique_times) == 1:
        if unique_times[0] == 0:
            return float(values.sum())
        count = min(len(values), int(processing_capacity_minutes // unique_times[0]))
        return float(np.sort(values)[-count:].sum()) if count > 0 else 0.0

    from scipy.optimize import Bounds, LinearConstraint, milp

    result = milp(
        c=-values,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(times, -np.inf, processing_capacity_minutes),
        options={"time_limit": 30, "mip_rel_gap": 0.0},
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"Exact allocation failed: {result.message}")
    return float(-result.fun)


def optimality_gap_percent(greedy_value: float, exact_value: float) -> float:
    """Return a nonnegative gap, failing if the alleged optimum is worse."""
    greedy_value = _finite(greedy_value, "greedy_value", minimum=0.0)
    exact_value = _finite(exact_value, "exact_value", minimum=0.0)
    tolerance = 1e-9 * max(1.0, abs(greedy_value), abs(exact_value))
    if greedy_value > exact_value + tolerance:
        raise RuntimeError(
            "Greedy objective exceeds exact objective: "
            f"greedy={greedy_value:.12g}, exact={exact_value:.12g}"
        )
    if exact_value <= tolerance:
        if greedy_value <= tolerance:
            return 0.0
        raise RuntimeError(
            "Exact objective is zero while greedy objective is nonzero: "
            f"greedy={greedy_value:.12g}, exact={exact_value:.12g}"
        )
    return max(0.0, 100.0 * (exact_value - greedy_value) / exact_value)


def _component_gross(
    asset: dict,
    prices: dict[str, float],
    base_yields: dict[str, float],
    rng: np.random.Generator,
) -> float:
    if "realized_component_value" in asset:
        return float(asset["realized_component_value"])
    return realize_value(
        asset["components"], prices, asset["true_yield_factor"], base_yields, rng
    )


def _recovery_result(
    asset: dict,
    gross_value: float,
    action: str,
    disposition: str,
    processing_costs: dict[str, dict],
) -> dict:
    proc = processing_costs[action]
    rework_cost, rework_time = rework_adjustment(asset, processing_costs)
    return {
        "realized_value": float(gross_value) - float(proc["cost"]) - rework_cost,
        "inspection_cost": asset["inspection"].cost,
        "inspection_level": asset["inspection"].level,
        "disposition": disposition,
        "time_min": asset["inspection"].time_min + float(proc["time_min"]) + rework_time,
    }


def _core_action_result(
    asset: dict,
    prices: dict[str, float],
    base_yields: dict[str, float],
    processing_costs: dict[str, dict],
    rng: np.random.Generator,
) -> dict:
    action = asset["best_level"]
    if action == "l0_refurb":
        gross = (
            float(processing_costs["refurb_values"][asset["asset_type"]])
            * float(asset["true_yield_factor"])
        )
        disposition = "refurbish"
    else:
        gross = _component_gross(asset, prices, base_yields, rng)
        disposition = "component_recovery"
    return _recovery_result(asset, gross, action, disposition, processing_costs)


def greedy_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict[str, float],
    base_yields: dict[str, float],
    processing_costs: dict[str, dict],
    rng: np.random.Generator,
    disable_ranking: bool = False,
) -> list[dict]:
    """Allocate labor with a positive-margin value-density heuristic."""
    score_assets(assets, prices, base_yields, processing_costs)
    processing_capacity = remaining_processing_capacity(assets, capacity_minutes)
    _, selected = greedy_expected_objective(
        assets, processing_capacity, disable_ranking=disable_ranking
    )
    return [
        _core_action_result(asset, prices, base_yields, processing_costs, rng)
        if id(asset) in selected else _scrap_result(asset)
        for asset in assets
    ]


def random_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict[str, float],
    base_yields: dict[str, float],
    processing_costs: dict[str, dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Randomly assign component, partial-recovery, or scrap actions."""
    score_assets(assets, prices, base_yields, processing_costs)
    processing_capacity = remaining_processing_capacity(assets, capacity_minutes)
    used = 0.0
    results = [_scrap_result(asset) for asset in assets]
    order = list(range(len(assets)))
    rng.shuffle(order)
    fraction = float(processing_costs["l1"]["recovery_fraction"])

    for idx in order:
        action_index = int(rng.integers(0, 3))
        if action_index == 2:
            continue
        action = "l2" if action_index == 0 else "l1"
        action_time = float(processing_costs[action]["time_min"]) - float(
            processing_costs["scrap"]["time_min"]
        )
        if used + action_time > processing_capacity + 1e-9:
            continue
        used += action_time
        gross = _component_gross(assets[idx], prices, base_yields, rng)
        if action == "l1":
            gross *= fraction
        disposition = "component_recovery" if action == "l2" else "partial_recovery"
        results[idx] = _recovery_result(
            assets[idx], gross, action, disposition, processing_costs
        )
    return results


def fifo_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict[str, float],
    base_yields: dict[str, float],
    processing_costs: dict[str, dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Choose the best positive action but admit assets strictly in arrival order."""
    return greedy_allocate(
        assets, capacity_minutes, prices, base_yields, processing_costs, rng,
        disable_ranking=True,
    )


def threshold_allocate(
    assets: list[dict],
    capacity_minutes: float,
    prices: dict[str, float],
    base_yields: dict[str, float],
    processing_costs: dict[str, dict],
    rng: np.random.Generator,
    routing_thresholds: dict[str, float],
) -> list[dict]:
    """Route by fixed condition thresholds without value-density ranking."""
    score_assets(assets, prices, base_yields, processing_costs)
    processing_capacity = remaining_processing_capacity(assets, capacity_minutes)
    used = 0.0
    results = []
    recover_at = float(routing_thresholds["recover"])
    partial_at = float(routing_thresholds["partial"])
    partial_fraction = float(processing_costs["l1"]["recovery_fraction"])

    for asset in assets:
        if asset["phi"] > recover_at:
            action = asset["best_level"]
            expected_increment = asset["processing_margin"]
        elif asset["phi"] > partial_at:
            action = "l1"
            expected_increment = (
                asset["expected_component_value"] * partial_fraction
                - float(processing_costs[action]["cost"])
                + float(processing_costs["scrap"]["cost"])
            )
        else:
            results.append(_scrap_result(asset))
            continue

        action_time = float(processing_costs[action]["time_min"]) - float(
            processing_costs["scrap"]["time_min"]
        )
        if expected_increment <= 0 or used + action_time > processing_capacity + 1e-9:
            results.append(_scrap_result(asset))
            continue
        used += action_time
        if action == "l0_refurb":
            gross = (
                float(processing_costs["refurb_values"][asset["asset_type"]])
                * float(asset["true_yield_factor"])
            )
            disposition = "refurbish"
        else:
            gross = _component_gross(asset, prices, base_yields, rng)
            disposition = "component_recovery"
            if action == "l1":
                gross *= partial_fraction
                disposition = "partial_recovery"
        results.append(_recovery_result(
            asset, gross, action, disposition, processing_costs
        ))
    return results
