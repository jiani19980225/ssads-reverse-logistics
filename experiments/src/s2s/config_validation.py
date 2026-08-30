"""Validation for the synthetic benchmark configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _number(value, path: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{path} must be finite")
    if minimum is not None and out < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    if maximum is not None and out > maximum:
        raise ValueError(f"{path} must be at most {maximum}")
    return out


def _mapping(value, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _cost_time_entry(value, path: str) -> None:
    entry = _mapping(value, path)
    for field in ("cost", "time_min"):
        if field not in entry:
            raise ValueError(f"{path}.{field} is required")
        _number(entry[field], f"{path}.{field}", minimum=0.0)


def validate_config(config: dict) -> dict:
    """Reject malformed or internally inconsistent benchmark assumptions."""
    _mapping(config, "config")
    required = {
        "name", "n_assets", "asset_types", "prices", "base_yields",
        "inspection_costs", "inspection_observation_noise",
        "inspection_update_weights", "processing_costs", "capacity",
        "thresholds", "routing_thresholds", "note_distribution",
        "condition_yield_ranges", "note_noise", "realized_yield_concentration",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")

    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("name must be a non-empty string")
    n_assets = config["n_assets"]
    if isinstance(n_assets, bool) or not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer")

    prices = _mapping(config["prices"], "prices")
    yields = _mapping(config["base_yields"], "base_yields")
    if not prices:
        raise ValueError("prices must not be empty")
    if set(prices) != set(yields):
        raise ValueError("prices and base_yields must have identical component keys")
    for name, price in prices.items():
        _number(price, f"prices.{name}", minimum=0.0)
        _number(yields[name], f"base_yields.{name}", minimum=0.0, maximum=1.0)

    asset_types = config["asset_types"]
    if not isinstance(asset_types, list) or not asset_types:
        raise ValueError("asset_types must be a non-empty list")
    names = []
    total_weight = 0.0
    for idx, raw_entry in enumerate(asset_types):
        entry = _mapping(raw_entry, f"asset_types[{idx}]")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"asset_types[{idx}].name must be a non-empty string")
        names.append(name)
        total_weight += _number(entry.get("weight"), f"asset_types[{idx}].weight", minimum=0.0)
        components = _mapping(entry.get("components"), f"asset_types[{idx}].components")
        if not components:
            raise ValueError(f"asset_types[{idx}].components must not be empty")
        for component, count in components.items():
            if component not in prices:
                raise ValueError(f"Unknown component {component!r} in asset type {name!r}")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"Component count for {name}.{component} must be a positive integer")
    if len(names) != len(set(names)):
        raise ValueError("asset type names must be unique")
    if total_weight <= 0:
        raise ValueError("asset type weights must sum to a positive value")

    inspection_costs = _mapping(config["inspection_costs"], "inspection_costs")
    for level in ("skip", "l1", "l2"):
        if level not in inspection_costs:
            raise ValueError(f"inspection_costs.{level} is required")
        _cost_time_entry(inspection_costs[level], f"inspection_costs.{level}")

    observation_noise = _mapping(
        config["inspection_observation_noise"], "inspection_observation_noise"
    )
    update_weights = _mapping(
        config["inspection_update_weights"], "inspection_update_weights"
    )
    for level in ("l1", "l2"):
        _number(observation_noise.get(f"{level}_sd"),
                f"inspection_observation_noise.{level}_sd", minimum=0.0)
        _number(update_weights.get(level), f"inspection_update_weights.{level}",
                minimum=0.0, maximum=1.0)

    processing = _mapping(config["processing_costs"], "processing_costs")
    for action in ("l1", "l2", "scrap"):
        if action not in processing:
            raise ValueError(f"processing_costs.{action} is required")
        _cost_time_entry(processing[action], f"processing_costs.{action}")
    _number(processing["l1"].get("recovery_fraction"),
            "processing_costs.l1.recovery_fraction", minimum=0.0, maximum=1.0)
    if "l0_refurb" in processing:
        _cost_time_entry(processing["l0_refurb"], "processing_costs.l0_refurb")
    for action in ("l1", "l2", "l0_refurb"):
        if action in processing and (
            float(processing[action]["time_min"]) < float(processing["scrap"]["time_min"])
        ):
            raise ValueError(f"processing_costs.{action}.time_min cannot be below scrap time")
    if "rework" in processing:
        _cost_time_entry(processing["rework"], "processing_costs.rework")
        _number(processing["rework"].get("trigger_yield_below"),
                "processing_costs.rework.trigger_yield_below", minimum=0.0, maximum=1.0)
    refurb_values = _mapping(processing.get("refurb_values", {}),
                             "processing_costs.refurb_values")
    unknown_refurb = set(refurb_values) - set(names)
    if unknown_refurb:
        raise ValueError(f"Unknown refurb asset types: {sorted(unknown_refurb)}")
    for name, value in refurb_values.items():
        _number(value, f"processing_costs.refurb_values.{name}", minimum=0.0)
    if refurb_values and "l0_refurb" not in processing:
        raise ValueError("refurb_values requires processing_costs.l0_refurb")

    capacity = _mapping(config["capacity"], "capacity")
    capacity_keys = [key for key in ("weekly_hours", "daily_hours") if key in capacity]
    if len(capacity_keys) != 1:
        raise ValueError("capacity must define exactly one of weekly_hours or daily_hours")
    capacity_value = _number(
        capacity[capacity_keys[0]], f"capacity.{capacity_keys[0]}", minimum=0.0
    )
    if capacity_value == 0:
        raise ValueError("capacity hours must be positive")

    thresholds = _mapping(config["thresholds"], "thresholds")
    tau_l = _number(thresholds.get("tau_l"), "thresholds.tau_l", minimum=0.0, maximum=1.0)
    tau_h = _number(thresholds.get("tau_h"), "thresholds.tau_h", minimum=0.0, maximum=1.0)
    if tau_l > tau_h:
        raise ValueError("thresholds.tau_l cannot exceed thresholds.tau_h")
    routing = _mapping(config["routing_thresholds"], "routing_thresholds")
    partial = _number(routing.get("partial"), "routing_thresholds.partial",
                      minimum=0.0, maximum=1.0)
    recover = _number(routing.get("recover"), "routing_thresholds.recover",
                      minimum=0.0, maximum=1.0)
    if partial > recover:
        raise ValueError("routing_thresholds.partial cannot exceed recover")

    noise = _mapping(config["note_noise"], "note_noise")
    p_omit = _number(noise.get("p_omit"), "note_noise.p_omit",
                     minimum=0.0, maximum=1.0)
    p_mislabel = _number(noise.get("p_mislabel"), "note_noise.p_mislabel",
                         minimum=0.0, maximum=1.0)
    if p_omit + p_mislabel > 1.0:
        raise ValueError("note_noise probabilities cannot sum above 1")
    concentration = _number(
        config["realized_yield_concentration"],
        "realized_yield_concentration",
        minimum=0.0,
    )
    if concentration == 0:
        raise ValueError("realized_yield_concentration must be positive")

    distribution = _mapping(config["note_distribution"], "note_distribution")
    ranges = _mapping(config["condition_yield_ranges"], "condition_yield_ranges")
    if not distribution:
        raise ValueError("note_distribution must not be empty")
    if set(distribution) != set(ranges):
        raise ValueError(
            "note_distribution and condition_yield_ranges must have identical keys"
        )
    total = sum(_number(value, f"note_distribution.{name}", minimum=0.0)
                for name, value in distribution.items())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("note_distribution probabilities must sum to 1")
    for name, bounds in ranges.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"condition_yield_ranges.{name} must contain [low, high]")
        low = _number(bounds[0], f"condition_yield_ranges.{name}[0]",
                      minimum=0.0, maximum=1.0)
        high = _number(bounds[1], f"condition_yield_ranges.{name}[1]",
                       minimum=0.0, maximum=1.0)
        if low > high:
            raise ValueError(f"condition_yield_ranges.{name} low cannot exceed high")
    return config
