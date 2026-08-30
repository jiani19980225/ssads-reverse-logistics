"""Smoke test: S1 with N=10 assets.

Scope: structural/reproducibility checks only. These run on a tiny N=10 batch
for speed, so they deliberately do not assert an economic ranking between methods.
Those comparisons are generated at scale by scripts/run_summary.py.

Validates:
1. Pipeline runs end-to-end without error
2. Output has correct structure (no NaN, positive n_assets)
3. Same seed produces identical results (reproducibility)
4. Different seeds produce different results (not degenerate)
5. Inspection accounting is complete before allocation
6. Analytical and exact-allocation behavior
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.decision_engine import (
    apply_rework_cost,
    compute_expected_value,
    exact_expected_objective,
    greedy_expected_objective,
)
from src.s2s.extractors.base import NullExtractor
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.extractors.strong import StrongExtractor
from src.s2s.metrics import RunMetrics
from src.s2s.pipeline import load_config, run_pipeline
from src.s2s.randomness import rng_for


@pytest.fixture
def s1_config():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "s1_it.yaml")
    cfg["n_assets"] = 10  # small for speed
    return cfg


@pytest.fixture
def s1_extractor():
    return KeywordExtractor("s1")


class TestDecisionModel:
    def test_phi_shifts_analytical_value(self):
        high, _ = compute_expected_value({"cpu": 1}, {"cpu": 100}, {"cpu": 0.8}, 0.9)
        low, _ = compute_expected_value({"cpu": 1}, {"cpu": 100}, {"cpu": 0.8}, 0.2)
        assert high > low

    def test_exact_objective_bounds_greedy(self):
        assets = [
            {"processing_margin": 60.0, "proc_time": 10.0, "value_density": 6.0},
            {"processing_margin": 100.0, "proc_time": 20.0, "value_density": 5.0},
            {"processing_margin": 120.0, "proc_time": 30.0, "value_density": 4.0},
        ]
        greedy, _ = greedy_expected_objective(assets, 40.0)
        exact = exact_expected_objective(assets, 40.0)
        assert exact >= greedy

    def test_rework_cost_is_allocator_independent(self):
        asset = {
            "inspection": type("Inspection", (), {"level": 0})(),
            "true_yield_factor": 0.20,
        }
        costs = {"rework": {"cost": 30, "time_min": 8, "trigger_yield_below": 0.30}}
        assert apply_rework_cost(asset, 100.0, costs) == 70.0


class TestPipeline:
    def test_runs_end_to_end(self, s1_config, s1_extractor):
        """Pipeline completes without error."""
        result = run_pipeline(s1_config, s1_extractor, seed=42)
        assert isinstance(result, RunMetrics)
        assert result.n_assets == 10

    def test_no_nan(self, s1_config, s1_extractor):
        """No NaN in any metric."""
        result = run_pipeline(s1_config, s1_extractor, seed=42)
        assert not np.isnan(result.TRV)
        assert not np.isnan(result.RPR)
        assert not np.isnan(result.ICS)
        assert not np.isnan(result.TPR)

    def test_reproducibility(self, s1_config, s1_extractor):
        """Same seed -> identical results."""
        r1 = run_pipeline(s1_config, s1_extractor, seed=42)
        r2 = run_pipeline(s1_config, s1_extractor, seed=42)
        assert r1.TRV == r2.TRV
        assert r1.RPR == r2.RPR

    def test_different_seeds(self, s1_config, s1_extractor):
        """Different seeds -> different results (not degenerate)."""
        r1 = run_pipeline(s1_config, s1_extractor, seed=42)
        r2 = run_pipeline(s1_config, s1_extractor, seed=123)
        # With 10 assets, results should differ. Compare the full metric tuple,
        # since any single metric could coincide across two seeds.
        assert (r1.TRV, r1.RPR, r1.ICS) != (r2.TRV, r2.RPR, r2.ICS)

    def test_full_inspection_deterministic(self, s1_config):
        null = NullExtractor()
        r1 = run_pipeline(s1_config, null, seed=42, inspection_mode="full")
        r2 = run_pipeline(s1_config, null, seed=42, inspection_mode="full")
        assert r1.TRV == r2.TRV
        assert r1.ICS == r2.ICS

    def test_all_inspections_are_charged_before_allocation(self, s1_config):
        # 10 hours of inspection plus 50 minutes of default scrap handling.
        s1_config["capacity"]["weekly_hours"] = 11
        result = run_pipeline(
            s1_config,
            NullExtractor(),
            seed=42,
            inspection_mode="full",
        )
        assert result.RPR == 0
        assert result.inspection_full == 10
        assert result.ICS == 0
        assert result.TRV == -10 * s1_config["inspection_costs"]["l2"]["cost"]


class TestSeedControl:
    """Locks the seed guarantees that make paired Wilcoxon valid (PROBLEM 4)."""

    def _population(self, cfg, seed):
        from src.data_generators.common import generate_assets
        gen_rng = rng_for(seed, "generation")
        assets = generate_assets(cfg, gen_rng)
        return [(a["asset_type"], round(a["true_yield_factor"], 9), a["text"]) for a in assets]

    def test_same_seed_shared_population(self, s1_config):
        """Same seed -> identical population (every method sees the same assets)."""
        assert self._population(s1_config, 7) == self._population(s1_config, 7)

    def test_different_seeds_independent_population(self, s1_config):
        """Different seeds -> genuinely different populations, not a reshuffle."""
        p0 = self._population(s1_config, 0)
        p1 = self._population(s1_config, 1)
        assert p0 != p1
        # sorted true_yields differ -> not the same population re-ordered
        assert sorted(t[1] for t in p0) != sorted(t[1] for t in p1)


class TestKeywordExtractor:
    def test_same_note_is_deterministic(self):
        ext = KeywordExtractor("s1")
        note = "Visible burn marks. Thermal damage."
        assert ext.extract(note, np.random.default_rng(1)) == ext.extract(
            note, np.random.default_rng(999)
        )

    def test_negative_signals_low_phi(self):
        ext = KeywordExtractor("s1")
        rng = np.random.default_rng(0)
        result = ext.extract("Visible burn marks. Thermal damage.", rng)
        assert result.phi <= 0.40  # multiplicative: clips at 0.40 max
        assert result.sigma > 0.7

    def test_positive_signals_high_phi(self):
        ext = KeywordExtractor("s1")
        rng = np.random.default_rng(0)
        result = ext.extract("Routine decommission. Clean. No corrosion.", rng)
        assert result.phi > 0.75
        assert result.sigma > 0.7

    def test_no_signals_default(self):
        ext = KeywordExtractor("s1")
        rng = np.random.default_rng(0)
        result = ext.extract("Asset received.", rng)
        assert result.phi == 1.0
        assert result.sigma < 0.3


class TestStrongExtractor:
    def test_s2_crack_is_not_misread_as_serviceable(self):
        ext = StrongExtractor("s2")
        result = ext.extract(
            "Crack found in engine mount during borescope. Exceeds serviceable limits.",
            np.random.default_rng(0),
        )
        assert result.phi == pytest.approx(0.25)
        assert result.sigma == pytest.approx(0.90)


class _FakeOpenAI:
    """Offline stand-in for the OpenAI client (used by DeepSeekExtractor)."""

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _FakeOpenAI._Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_FakeOpenAI._Choice(content)]

    class _Completions:
        def __init__(self, payload):
            self._payload = payload

        def create(self, **kwargs):
            import json
            return _FakeOpenAI._Resp(json.dumps(self._payload))

    class _Chat:
        def __init__(self, payload):
            self.completions = _FakeOpenAI._Completions(payload)

    def __init__(self, payload):
        self.chat = _FakeOpenAI._Chat(payload)


class TestDeepSeekExtractor:
    def test_parses_valid_response(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        fake = _FakeOpenAI({"phi": 0.08, "sigma": 0.92, "condition": "damaged"})
        ext = DeepSeekExtractor("s1", client=fake)
        result = ext.extract("PSU failure. Burn marks on mainboard.", None)
        assert result.phi == pytest.approx(0.08, abs=0.01)
        assert result.sigma == pytest.approx(0.92, abs=0.01)

    def test_rejects_out_of_range_live_response(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor

        fake = _FakeOpenAI({"phi": 1.2, "sigma": 0.5, "condition": "clean"})
        ext = DeepSeekExtractor("s1", client=fake)
        with pytest.raises(ValueError, match="valid phi and sigma"):
            ext.extract("Clean unit.", None)

    @pytest.mark.parametrize("phi", [True, "0.9"])
    def test_rejects_non_numeric_live_response(self, phi):
        from src.s2s.extractors.deepseek import DeepSeekExtractor

        fake = _FakeOpenAI({"phi": phi, "sigma": 0.5, "condition": "clean"})
        ext = DeepSeekExtractor("s1", client=fake)
        with pytest.raises(ValueError, match="valid phi and sigma"):
            ext.extract("Clean unit.", None)

    def test_cache_hit_skips_api(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        cache = {"Clean unit. No issues.": (0.95, 0.90)}

        class _Boom:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise AssertionError("should not reach API on cache hit")

        ext = DeepSeekExtractor("s1", client=_Boom(), response_cache=cache)
        r = ext.extract("Clean unit. No issues.", None)
        assert r.phi == pytest.approx(0.95)

    def test_invalid_cache_entry_is_rejected(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor

        ext = DeepSeekExtractor("s1", response_cache={"bad": (float("nan"), 0.5)})
        with pytest.raises(ValueError, match="finite"):
            ext.extract("bad", None)

    def test_non_numeric_cache_entry_is_rejected(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor

        ext = DeepSeekExtractor("s1", response_cache={"bad": ("0.9", 0.5)})
        with pytest.raises(TypeError, match="numeric"):
            ext.extract("bad", None)

    def test_missing_key_raises(self, monkeypatch):
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        ext = DeepSeekExtractor("s2")
        with pytest.raises(RuntimeError):
            ext.extract("Corroded panel.", None)

    def test_cache_only_mode_rejects_miss_before_client_call(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor

        class _Boom:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise AssertionError("cache-only mode must not call the API")

        ext = DeepSeekExtractor("s1", client=_Boom(), allow_live=False)
        with pytest.raises(RuntimeError, match="cache-only mode"):
            ext.extract("Previously unseen note.", None)

    def test_unknown_scenario_rejected(self):
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        with pytest.raises(ValueError):
            DeepSeekExtractor("s9")
