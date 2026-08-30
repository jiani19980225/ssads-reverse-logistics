"""Adversarial tests for action choice, capacity accounting, and paired outcomes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.decision_engine import (
    CapacityInfeasibleError,
    compute_expected_value,
    exact_expected_objective,
    greedy_expected_objective,
    realize_value,
    remaining_processing_capacity,
    score_assets,
)
from src.s2s.extractors.base import NullExtractor
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.inspection_policy import InspectionDecision
from src.s2s.pipeline import load_config, run_pipeline


@pytest.fixture
def s1_config():
    config = load_config(Path(__file__).parent.parent / "configs" / "s1_it.yaml")
    config["n_assets"] = 20
    return config


def test_action_choice_uses_net_margin_not_gross_value():
    costs: dict[str, dict] = {
        "l2": {"cost": 50, "time_min": 10},
        "l0_refurb": {"cost": 0, "time_min": 10},
    }
    gross, action = compute_expected_value(
        {"part": 1}, {"part": 100}, {"part": 1.0}, 1.0,
        refurb_value=90, processing_costs=costs,
    )
    assert gross == 90
    assert action == "l0_refurb"


def test_realize_value_supports_one_component():
    value = realize_value(
        {"part": 1}, {"part": 100}, 0.5, {"part": 0.8},
        np.random.default_rng(3),
    )
    assert np.isfinite(value)
    assert value >= 0


def test_realize_value_respects_zero_baseline_yield():
    value = realize_value(
        {"part": 1}, {"part": 100}, 0.5, {"part": 0.0},
        np.random.default_rng(3),
    )
    assert value == 0.0


def test_scrap_handling_is_reserved_before_recovery():
    assets = [
        {"inspection": InspectionDecision(0, 0, 0)},
        {"inspection": InspectionDecision(0, 0, 0)},
    ]
    costs = {"scrap": {"cost": 0, "time_min": 5}}
    assert remaining_processing_capacity(assets, 10, costs) == 0
    with pytest.raises(CapacityInfeasibleError):
        remaining_processing_capacity(assets, 9, costs)


def test_exact_objective_can_improve_nonuniform_greedy_choice():
    assets = [
        {"processing_margin": 60.0, "capacity_time": 10.0, "proc_time": 10.0,
         "value_density": 6.0},
        {"processing_margin": 100.0, "capacity_time": 20.0, "proc_time": 20.0,
         "value_density": 5.0},
        {"processing_margin": 120.0, "capacity_time": 30.0, "proc_time": 30.0,
         "value_density": 4.0},
    ]
    greedy, _ = greedy_expected_objective(assets, 40)
    exact = exact_expected_objective(assets, 40)
    assert greedy == 160
    assert exact == 180


def test_same_seed_predraws_same_asset_outcomes_across_methods(s1_config):
    _, no_signal_assets, _ = run_pipeline(
        s1_config, NullExtractor(), 7, inspection_mode="none", return_details=True
    )
    _, semantic_assets, _ = run_pipeline(
        s1_config, KeywordExtractor("s1"), 7, inspection_mode="policy", return_details=True
    )
    assert [a["realized_component_value"] for a in no_signal_assets] == [
        a["realized_component_value"] for a in semantic_assets
    ]


def test_fifo_matches_no_signal_when_capacity_does_not_bind(s1_config):
    no_signal = run_pipeline(
        s1_config, NullExtractor(), 8, inspection_mode="none", allocator="greedy"
    )
    fifo = run_pipeline(
        s1_config, NullExtractor(), 8, inspection_mode="none", allocator="fifo"
    )
    assert fifo.TRV == no_signal.TRV
    assert fifo.RPR == no_signal.RPR == 1.0


def test_unknown_allocator_is_rejected(s1_config):
    with pytest.raises(ValueError, match="Unknown allocator"):
        run_pipeline(s1_config, NullExtractor(), 1, allocator="typo")


def test_rework_time_and_cost_are_in_realized_result():
    config = load_config(Path(__file__).parent.parent / "configs" / "s3_consumer.yaml")
    config["n_assets"] = 1

    def one_asset(_config, _rng):
        return [{
            "asset_id": 0,
            "asset_type": "smartphone",
            "age_bracket": 0,
            "components": {"pcb": 1, "battery": 1, "display": 1, "camera": 2},
            "text": "No detail.",
            "true_condition": "dead",
            "observed_condition": "uninformative",
            "true_yield_factor": 0.20,
        }]

    metrics, _, results = run_pipeline(
        config, NullExtractor(), 1, inspection_mode="none",
        asset_generator=one_asset, return_details=True,
    )
    result = results[0]
    assert result["disposition"] == "refurbish"
    assert result["time_min"] == 13
    assert metrics.TRV == pytest.approx(120 * 0.20 - 8 - 30)


def test_score_assets_accounts_for_scrap_cost_and_incremental_time():
    asset = {
        "asset_type": "unit",
        "components": {"part": 1},
        "phi": 1.0,
    }
    costs: dict[str, dict] = {
        "l1": {"cost": 1, "time_min": 4, "recovery_fraction": 0.6},
        "l2": {"cost": 20, "time_min": 10},
        "scrap": {"cost": 5, "time_min": 3},
    }
    score_assets([asset], {"part": 100}, {"part": 1.0}, costs)
    assert asset["processing_margin"] == 85
    assert asset["capacity_time"] == 7
