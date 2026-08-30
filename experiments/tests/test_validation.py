"""Boundary tests for configuration, extraction, and random stream contracts."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generators.common import generate_assets
from src.data_generators.noise import observed_condition, resolve_noise
from src.s2s.beta_model import BetaParams, ground_truth_params, sample_yield
from src.s2s.config_validation import validate_config
from src.s2s.decision_engine import optimality_gap_percent
from src.s2s.extractors.base import ExtractionResult, NullExtractor
from src.s2s.extractors.strong import StrongExtractor
from src.s2s.inspection_policy import adaptive_inspection
from src.s2s.metrics import compute_metrics
from src.s2s.pipeline import load_config, run_pipeline
from src.s2s.randomness import parse_seed_spec, rng_for


@pytest.fixture
def config():
    return load_config(Path(__file__).parent.parent / "configs" / "s1_it.yaml")


def test_endpoint_perturbation_is_clipped_or_moves_inward():
    rng = np.random.default_rng(1)
    observed = []
    for _ in range(100):
        observed.append(observed_condition(
            "best", ["best", "middle", "worst"], rng, 0.0, 1.0, "unknown"
        ))
    assert set(observed) == {"best", "middle"}
    assert 25 < observed.count("middle") < 75


def test_paired_noise_path_consumes_fixed_random_draws():
    rng_no_noise = np.random.default_rng(11)
    rng_high_noise = np.random.default_rng(11)
    observed_condition(
        "middle", ["best", "middle", "worst"], rng_no_noise,
        0.0, 0.0, "unknown", fixed_draw_count=True,
    )
    observed_condition(
        "middle", ["best", "middle", "worst"], rng_high_noise,
        0.3, 0.4, "unknown", fixed_draw_count=True,
    )
    assert np.array_equal(rng_no_noise.random(5), rng_high_noise.random(5))


def test_noise_probabilities_cannot_exceed_one(config):
    bad = copy.deepcopy(config)
    bad["note_noise"] = {"p_omit": 0.7, "p_mislabel": 0.4}
    with pytest.raises(ValueError, match="sum above 1"):
        validate_config(bad)
    with pytest.raises(ValueError, match="sum above 1"):
        resolve_noise(bad)


def test_config_rejects_missing_component_price(config):
    bad = copy.deepcopy(config)
    del bad["prices"]["cpu"]
    with pytest.raises(ValueError, match="identical component keys"):
        validate_config(bad)


def test_config_rejects_inconsistent_condition_ranges(config):
    bad = copy.deepcopy(config)
    del bad["condition_yield_ranges"]["clean"]
    with pytest.raises(ValueError, match="identical keys"):
        validate_config(bad)

    bad = copy.deepcopy(config)
    bad["condition_yield_ranges"]["clean"] = [0.9, 0.8]
    with pytest.raises(ValueError, match="low cannot exceed high"):
        validate_config(bad)


def test_config_rejects_condition_probabilities_not_summing_to_one(config):
    bad = copy.deepcopy(config)
    bad["note_distribution"]["clean"] = 0.30
    with pytest.raises(ValueError, match="sum to 1"):
        validate_config(bad)


def test_config_rejects_nonpositive_realized_yield_concentration(config):
    bad = copy.deepcopy(config)
    bad["realized_yield_concentration"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        validate_config(bad)


@pytest.mark.parametrize("phi,sigma", [(0.0, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, float("nan"))])
def test_extraction_result_rejects_invalid_scores(phi, sigma):
    with pytest.raises((TypeError, ValueError)):
        ExtractionResult(phi=phi, sigma=sigma)


def test_inspection_threshold_order_is_validated(config):
    with pytest.raises(ValueError, match="tau_l"):
        adaptive_inspection(0.5, 0.2, 0.8, config["inspection_costs"])


def test_sample_yield_preserves_single_component_dimension():
    params = ground_truth_params(np.array([0.5]))
    draw = sample_yield(params, np.random.default_rng(2))
    assert draw.shape == (1,)
    draws = sample_yield(params, np.random.default_rng(2), n=3)
    assert draws.shape == (3, 1)


def test_sample_yield_rejects_invalid_beta_parameters():
    with pytest.raises(ValueError, match="positive"):
        sample_yield(BetaParams(alpha=np.array([0.0]), beta=np.array([1.0])),
                     np.random.default_rng(2))


@pytest.mark.parametrize(
    "true_yield", [-0.01, 1.01, float("inf"), "not-a-number", "0.5", True]
)
def test_ground_truth_params_rejects_invalid_yield(true_yield):
    with pytest.raises((TypeError, ValueError)):
        ground_truth_params(true_yield)


@pytest.mark.parametrize("concentration", [True, "20", 0, float("inf")])
def test_ground_truth_params_rejects_invalid_concentration(concentration):
    with pytest.raises((TypeError, ValueError)):
        ground_truth_params(0.5, concentration)


@pytest.mark.parametrize(
    "spec,expected",
    [("0-2", [0, 1, 2]), ("0, 2,5", [0, 2, 5])],
)
def test_seed_parser(spec, expected):
    assert parse_seed_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "3-1", "1,1", "a,b", "-1"])
def test_seed_parser_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        parse_seed_spec(spec)


def test_named_random_streams_are_repeatable_and_distinct():
    a = rng_for(4, "generation").random(5)
    b = rng_for(4, "generation").random(5)
    c = rng_for(4, "outcome").random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    "config_name",
    ["s1_it.yaml", "s2_aviation.yaml", "s3_consumer.yaml"],
)
def test_noise_variants_share_identical_latent_assets(config_name):
    cfg = load_config(Path(__file__).parent.parent / "configs" / config_name)
    cfg["n_assets"] = 25
    signatures = []
    for noise in (
        {"p_omit": 0.0, "p_mislabel": 0.0},
        {"p_omit": 0.3, "p_mislabel": 0.25},
        {"p_omit": 0.3, "p_mislabel": 0.4},
    ):
        variant = copy.deepcopy(cfg)
        variant["note_noise"] = noise
        assets = generate_assets(
            variant,
            rng_for(7, "generation"),
            rng_for(7, "note_sensitivity"),
        )
        signatures.append([
            (
                asset["asset_type"],
                asset["age_bracket"],
                asset["true_condition"],
                asset["true_yield_factor"],
            )
            for asset in assets
        ])
    assert signatures[0] == signatures[1] == signatures[2]


def test_pipeline_rejects_generator_output_inconsistent_with_config(config):
    config = copy.deepcopy(config)
    config["n_assets"] = 1

    def wrong_components(cfg, rng):
        assets = generate_assets(cfg, rng)
        assets[0]["components"] = {"cpu": 999}
        return assets

    with pytest.raises(ValueError, match="components do not match"):
        run_pipeline(config, NullExtractor(), 0, asset_generator=wrong_components)

    def wrong_yield(cfg, rng):
        assets = generate_assets(cfg, rng)
        condition = assets[0]["true_condition"]
        assets[0]["true_yield_factor"] = cfg["condition_yield_ranges"][condition][1] + 0.01
        return assets

    with pytest.raises(ValueError, match="configured range"):
        run_pipeline(config, NullExtractor(), 0, asset_generator=wrong_yield)


def test_dropping_a_keyword_family_falls_back_and_leaves_default_intact():
    from src.s2s.extractors.keyword import _SIGNALS, KeywordExtractor

    damaged = "PSU failure. Visible burn marks on mainboard. CPU smells burnt."
    default = KeywordExtractor("s1")
    assert default.extract(damaged, None).phi < 0.5

    dropped = KeywordExtractor("s1", drop_families=("negative",))
    result = dropped.extract(damaged, None)
    # An unwritten failure family matches nothing, so the reader falls back to
    # the maximally optimistic prior. This is the audited failure mode.
    assert result.used_fallback is True
    assert result.phi == 1.0
    # The shared vocabulary must not be mutated by constructing a dropped reader.
    assert _SIGNALS["s1"]["negative"]
    assert KeywordExtractor("s1").extract(damaged, None).phi < 0.5

    with pytest.raises(ValueError, match="Unknown signal families"):
        KeywordExtractor("s1", drop_families=("nonexistent",))


def test_exact_gap_audit_rejects_greedy_above_optimum():
    with pytest.raises(RuntimeError, match="Greedy objective exceeds exact"):
        optimality_gap_percent(101.0, 100.0)


def test_metrics_reject_unknown_disposition_and_zero_total_time():
    base = {
        "realized_value": 1.0,
        "inspection_cost": 0.0,
        "inspection_level": 0,
        "disposition": "scrap",
        "time_min": 1.0,
    }
    bad_disposition = {**base, "disposition": "unknown"}
    with pytest.raises(ValueError, match="invalid disposition"):
        compute_metrics([bad_disposition], 0.0)
    zero_time = {**base, "time_min": 0.0}
    with pytest.raises(ValueError, match="positive total processing time"):
        compute_metrics([zero_time], 0.0)


@pytest.mark.parametrize("capacity_fraction", [True, "1.0", None])
def test_capacity_fraction_must_be_numeric(config, capacity_fraction):
    with pytest.raises(TypeError, match="capacity_fraction must be numeric"):
        run_pipeline(config, NullExtractor(), 0, capacity_fraction=capacity_fraction)


@pytest.mark.parametrize(
    "config_name,scenario,expected_phi",
    [
        (
            "s1_it.yaml",
            "s1",
            {"clean": 0.92, "mixed": 0.60, "damaged": 0.18, "ambiguous": 0.50},
        ),
        (
            "s2_aviation.yaml",
            "s2",
            {
                "serviceable": 0.85,
                "worn": 0.55,
                "inoperative": 0.45,
                "corroded": 0.35,
                "cracked": 0.25,
                "failed": 0.15,
            },
        ),
        (
            "s3_consumer.yaml",
            "s3",
            {
                "functional_return": 0.95,
                "cosmetic": 0.82,
                "degraded": 0.52,
                "dead": 0.20,
                "hazard": 0.06,
            },
        ),
    ],
)
def test_strong_matcher_has_no_generated_template_collisions(
    config_name, scenario, expected_phi
):
    cfg = load_config(Path(__file__).parent.parent / "configs" / config_name)
    extractor = StrongExtractor(scenario)
    for seed in range(3):
        for asset in generate_assets(cfg, rng_for(seed, "generation")):
            observed = asset["observed_condition"]
            if observed in expected_phi:
                result = extractor.extract(asset["text"], None)
                assert result.phi == pytest.approx(expected_phi[observed])
