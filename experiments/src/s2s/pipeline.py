"""One complete synthetic benchmark run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, overload

import numpy as np
import yaml

from .config_validation import validate_config
from .decision_engine import (
    fifo_allocate,
    greedy_allocate,
    prepare_realized_outcomes,
    random_allocate,
    threshold_allocate,
)
from .extractors.base import AbstractExtractor
from .inspection_policy import (
    adaptive_inspection,
    inspect_all,
    inspection_at_level,
    skip_inspection,
)
from .metrics import RunMetrics, compute_metrics
from .randomness import rng_for


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def _random_levels(n_assets: int, counts: dict[int, int], rng: np.random.Generator) -> list[int]:
    if set(counts) - {0, 1, 2}:
        raise ValueError("Random inspection levels must be 0, 1, or 2")
    levels = []
    for level in (0, 1, 2):
        count = counts.get(level, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Random inspection-level counts must be nonnegative integers")
        levels.extend([level] * count)
    if len(levels) != n_assets:
        raise ValueError("Random inspection-level counts must sum to the batch size")
    rng.shuffle(levels)
    return levels


def apply_inspection_stage(
    assets: list[dict],
    config: dict,
    use_adaptive_inspection: bool,
    inspection_mode: str,
    inspect_rng: np.random.Generator,
    inspect_fraction: float | None = None,
    random_level_counts: dict[int, int] | None = None,
) -> None:
    """Assign, charge, and reveal all first-stage inspections.

    Quick and full inspection receive noisy condition measurements. Quick
    inspection gives the measurement 50 percent weight; full inspection gives
    it 90 percent. The explicit oracle mode reveals condition exactly.
    """
    costs = config["inspection_costs"]
    tau_h = config["thresholds"]["tau_h"]
    tau_l = config["thresholds"]["tau_l"]
    observation_noise = config["inspection_observation_noise"]
    update_weights = config["inspection_update_weights"]
    mode = inspection_mode
    if mode == "policy" and not use_adaptive_inspection:
        mode = "full"

    valid_modes = {"policy", "full", "oracle", "none", "random_budget", "random_levels"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown inspection mode: {mode}")

    # Pre-draw both physical-observation errors for every asset. All methods
    # therefore see the same potential measurement for a given asset and seed.
    l1_errors = inspect_rng.normal(0.0, observation_noise["l1_sd"], size=len(assets))
    l2_errors = inspect_rng.normal(0.0, observation_noise["l2_sd"], size=len(assets))

    random_full = set()
    random_levels = None
    if mode == "random_budget":
        if inspect_fraction is None:
            raise ValueError("random_budget requires inspect_fraction")
        if isinstance(inspect_fraction, bool) or not isinstance(inspect_fraction, (int, float)):
            raise TypeError("inspect_fraction must be numeric")
        if not math.isfinite(inspect_fraction) or not 0.0 <= inspect_fraction <= 1.0:
            raise ValueError("inspect_fraction must be in [0, 1]")
        k = round(inspect_fraction * len(assets))
        if k:
            random_full = set(
                inspect_rng.choice(len(assets), size=min(k, len(assets)), replace=False).tolist()
            )
    elif mode == "random_levels":
        if random_level_counts is None:
            raise ValueError("random_levels requires random_level_counts")
        random_levels = _random_levels(len(assets), random_level_counts, inspect_rng)

    for idx, asset in enumerate(assets):
        if mode == "policy":
            decision = adaptive_inspection(asset["sigma"], tau_h, tau_l, costs)
        elif mode in {"full", "oracle"}:
            decision = inspect_all(costs)
        elif mode == "none":
            decision = skip_inspection(costs)
        elif mode == "random_budget":
            decision = inspect_all(costs) if idx in random_full else skip_inspection(costs)
        elif mode == "random_levels":
            assert random_levels is not None
            decision = inspection_at_level(random_levels[idx], costs)
        else:
            raise AssertionError("validated inspection mode was not handled")
        asset["inspection"] = decision

        if mode == "oracle":
            asset["phi"] = asset["true_yield_factor"]
        elif decision.level == 2:
            observed = float(np.clip(
                asset["true_yield_factor"] + l2_errors[idx],
                0.01,
                1.0,
            ))
            weight = update_weights["l2"]
            asset["phi"] = (1.0 - weight) * asset["phi"] + weight * observed
        elif decision.level == 1:
            observed = float(np.clip(
                asset["true_yield_factor"] + l1_errors[idx],
                0.01,
                1.0,
            ))
            weight = update_weights["l1"]
            asset["phi"] = (1.0 - weight) * asset["phi"] + weight * observed


def _validate_assets(assets: list[dict], config: dict) -> None:
    expected_n = config["n_assets"]
    if not isinstance(assets, list) or len(assets) != expected_n:
        raise ValueError(f"Asset generator must return exactly {expected_n} assets")
    required = {
        "asset_id", "asset_type", "age_bracket", "components", "text",
        "true_condition", "observed_condition", "true_yield_factor",
    }
    type_components = {
        entry["name"]: dict(entry["components"])
        for entry in config["asset_types"]
    }
    condition_ranges = config["condition_yield_ranges"]
    ids: list[int] = []
    for idx, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise TypeError(f"Asset {idx} must be a dictionary")
        missing = sorted(required - set(asset))
        if missing:
            raise ValueError(f"Asset {idx} is missing fields: {', '.join(missing)}")
        asset_id = asset["asset_id"]
        if isinstance(asset_id, bool) or not isinstance(asset_id, int):
            raise TypeError(f"Asset {idx} asset_id must be an integer")
        ids.append(asset_id)
        asset_type = asset["asset_type"]
        if not isinstance(asset_type, str) or asset_type not in type_components:
            raise ValueError(f"Asset {idx} has unknown asset_type {asset_type!r}")
        if asset["components"] != type_components[asset_type]:
            raise ValueError(
                f"Asset {idx} components do not match configured type {asset_type!r}"
            )
        age_bracket = asset["age_bracket"]
        if isinstance(age_bracket, bool) or not isinstance(age_bracket, int):
            raise TypeError(f"Asset {idx} age_bracket must be an integer")
        if age_bracket not in (0, 1):
            raise ValueError(f"Asset {idx} age_bracket must be 0 or 1")
        if not isinstance(asset["text"], str) or not asset["text"].strip():
            raise ValueError(f"Asset {idx} text must be a non-empty string")
        true_condition = asset["true_condition"]
        if not isinstance(true_condition, str) or true_condition not in condition_ranges:
            raise ValueError(
                f"Asset {idx} has unknown true_condition {true_condition!r}"
            )
        if (
            not isinstance(asset["observed_condition"], str)
            or not asset["observed_condition"].strip()
        ):
            raise ValueError(
                f"Asset {idx} observed_condition must be a non-empty string"
            )
        value = asset["true_yield_factor"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Asset {idx} true_yield_factor must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Asset {idx} true_yield_factor must be finite")
        low, high = condition_ranges[true_condition]
        if not float(low) <= numeric_value <= float(high):
            raise ValueError(
                f"Asset {idx} true_yield_factor must be within the configured "
                f"range for {true_condition!r}"
            )
    if len(ids) != len(set(ids)):
        raise ValueError("asset_id values must be unique")


@overload
def run_pipeline(
    config: dict,
    extractor: AbstractExtractor,
    seed: int,
    use_adaptive_inspection: bool = True,
    capacity_fraction: float = 1.0,
    disable_greedy_ranking: bool = False,
    allocator: str = "greedy",
    asset_generator=None,
    inspection_mode: str = "policy",
    inspect_fraction: float | None = None,
    random_level_counts: dict[int, int] | None = None,
    return_details: Literal[False] = False,
) -> RunMetrics: ...


@overload
def run_pipeline(
    config: dict,
    extractor: AbstractExtractor,
    seed: int,
    use_adaptive_inspection: bool = True,
    capacity_fraction: float = 1.0,
    disable_greedy_ranking: bool = False,
    allocator: str = "greedy",
    asset_generator=None,
    inspection_mode: str = "policy",
    inspect_fraction: float | None = None,
    random_level_counts: dict[int, int] | None = None,
    return_details: Literal[True] = True,
) -> tuple[RunMetrics, list[dict], list[dict]]: ...


def run_pipeline(
    config: dict,
    extractor: AbstractExtractor,
    seed: int,
    use_adaptive_inspection: bool = True,
    capacity_fraction: float = 1.0,
    disable_greedy_ranking: bool = False,
    allocator: str = "greedy",
    asset_generator=None,
    inspection_mode: str = "policy",
    inspect_fraction: float | None = None,
    random_level_counts: dict[int, int] | None = None,
    return_details: bool = False,
) -> RunMetrics | tuple[RunMetrics, list[dict], list[dict]]:
    """Execute one run with isolated, paired random-number streams."""
    validate_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(capacity_fraction, bool) or not isinstance(capacity_fraction, (int, float)):
        raise TypeError("capacity_fraction must be numeric")
    if not math.isfinite(capacity_fraction) or capacity_fraction <= 0:
        raise ValueError("capacity_fraction must be finite and positive")
    gen_rng = rng_for(seed, "generation")
    extract_rng = rng_for(seed, "extraction")
    decision_rng = rng_for(seed, "allocation")
    inspect_rng = rng_for(seed, "inspection")
    outcome_rng = rng_for(seed, "outcome")

    if asset_generator is None:
        from ..data_generators.common import generate_assets

        assets = generate_assets(config, gen_rng)
    else:
        assets = asset_generator(config, gen_rng)

    _validate_assets(assets, config)
    prepare_realized_outcomes(
        assets,
        config["prices"],
        config["base_yields"],
        outcome_rng,
        concentration=config["realized_yield_concentration"],
    )

    for asset in assets:
        extraction = extractor.extract(asset["text"], extract_rng, asset=asset)
        asset["phi"] = extraction.phi
        asset["sigma"] = extraction.sigma

    apply_inspection_stage(
        assets,
        config,
        use_adaptive_inspection,
        inspection_mode,
        inspect_rng,
        inspect_fraction=inspect_fraction,
        random_level_counts=random_level_counts,
    )

    cap_key = "weekly_hours" if "weekly_hours" in config["capacity"] else "daily_hours"
    capacity_minutes = config["capacity"][cap_key] * 60 * capacity_fraction
    alloc_args = {
        "assets": assets,
        "capacity_minutes": capacity_minutes,
        "prices": config["prices"],
        "base_yields": config["base_yields"],
        "processing_costs": config["processing_costs"],
        "rng": decision_rng,
    }

    if allocator == "random":
        results = random_allocate(**alloc_args)
    elif allocator == "fifo":
        results = fifo_allocate(**alloc_args)
    elif allocator == "threshold":
        results = threshold_allocate(
            **alloc_args, routing_thresholds=config["routing_thresholds"]
        )
    elif allocator == "greedy":
        results = greedy_allocate(
            **alloc_args,
            disable_ranking=disable_greedy_ranking,
        )
    else:
        raise ValueError(f"Unknown allocator: {allocator}")

    metrics = compute_metrics(results, config["inspection_costs"]["l2"]["cost"])
    if return_details:
        return metrics, assets, results
    return metrics
