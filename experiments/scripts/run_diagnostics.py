"""Diagnostics for the paper's Error Analysis, Case Studies, and Sensitivity.

Reproduces the quantitative checks in "Error Analysis and Case Studies" and
"Sensitivity Analysis" that are not covered by the main summary:

  1. Overestimation (error) rate: share of assets the policy would skip on high
     signal quality (sigma >= tau_h) with a high context factor (phi > 0.7) whose
     true yield is nonetheless low (< 0.30) -- a high-score bad unit.
     Both the conditional rate and batch prevalence are printed.
  2. Capacity sensitivity: SSADS-Keyword TRV at capacity_fraction=0.5 vs 1.0.
     A scenario is reported as infeasible if planned first-stage inspections
     plus default scrap handling exceed the reduced labor budget.
  3. Case studies (S1): keyword-extractor phi/sigma on the two benchmark
     template notes quoted in the paper.

Every number is computed from the released generators/extractors; nothing is
hand-set. Usage:
    python scripts/run_diagnostics.py --seeds 0-29
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generators.common import generate_assets
from src.s2s.baselines.runner import run_baseline
from src.s2s.decision_engine import CapacityInfeasibleError
from src.s2s.extractors.base import ExtractionResult
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.pipeline import load_config
from src.s2s.randomness import parse_seed_spec, rng_for

SCENARIOS = [
    ("configs/s1_it.yaml", "s1", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "s2", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "s3", "S3: Consumer Electronics"),
]

HIGH_PHI_THRESHOLD = 0.70
BAD_YIELD_THRESHOLD = 0.30

# The two S1 template notes quoted as case studies in the paper.
CASE_NOTES = [
    ("Correct skip (clean)",
     "Routine decommission. All components seated properly. No corrosion. 4yr service."),
    (
        "Low-condition read (damaged)",
        (
            "PSU failure. Visible burn marks on mainboard near power connector J12. "
            "CPU smells burnt."
        ),
    ),
]


def _load_llm_cache(base_dir: Path, key: str) -> dict:
    """note text -> (phi, sigma) from the cached DeepSeek responses, or {}."""
    p = base_dir / "outputs" / "llm_cache" / f"{key}_deepseek_deepseek-chat.json"
    if not p.exists():
        return {}
    return {
        k: tuple(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()
    }


def error_rate(cfg: dict, key: str, seeds: list, llm_cache: dict | None = None) -> dict:
    """Overestimation error.

    A "high-score bad unit" is an asset the adaptive policy would skip on high
    signal quality (sigma >= tau_h) while the extractor reports a high context
    factor (phi > 0.7), yet true yield is low (< 0.30).

    We report two denominators because they answer different questions:
      - conditional rate = bad / (assets shipped on high signal quality)
            i.e. of the units we skip-and-keep on a high score, what fraction are bad.
            This is the rate quoted in the paper's Error Analysis.
      - prevalence       = bad / (all assets)
            i.e. how common such errors are across the whole batch.

    Seeding uses the same named generation and extraction streams as the simulation,
    so the asset stream is identical to every other number in the paper and to
    the corpus the LLM responses were cached on. With llm_cache supplied, phi/sigma
    are read from the cache (DeepSeek end-to-end) instead of the keyword extractor;
    any uncached note is counted as a miss and skipped.
    """
    ext = None if llm_cache is not None else KeywordExtractor(key)
    tau_h = cfg["thresholds"]["tau_h"]
    bad = high_score_skip = tot = miss = 0
    for seed in seeds:
        gen_rng = rng_for(seed, "generation")
        extract_rng = rng_for(seed, "extraction")
        for a in generate_assets(cfg, gen_rng):
            tot += 1
            if llm_cache is not None:
                if a["text"] not in llm_cache:
                    miss += 1
                    continue
                phi, sigma = llm_cache[a["text"]]
                cached = ExtractionResult(phi=float(phi), sigma=float(sigma))
                phi, sigma = cached.phi, cached.sigma
            else:
                assert ext is not None
                r = ext.extract(a["text"], extract_rng)
                phi, sigma = r.phi, r.sigma
            kept = sigma >= tau_h and phi > HIGH_PHI_THRESHOLD
            if kept:
                high_score_skip += 1
                if a["true_yield_factor"] < BAD_YIELD_THRESHOLD:
                    bad += 1
    evaluated = tot - miss
    return {
        "bad": bad,
        "high_score_skip": high_score_skip,
        "tot": tot,
        "evaluated": evaluated,
        "miss": miss,
        "rate_conditional": (
            bad / high_score_skip if high_score_skip else float("nan")
        ),
        "rate_prevalence": bad / evaluated if evaluated else float("nan"),
    }


def capacity_sensitivity(cfg: dict, key: str, seeds: list) -> tuple[float, float | None]:
    """SSADS-Keyword TRV at full vs half capacity (mean over seeds)."""
    ext = KeywordExtractor(key)
    full = np.mean([run_baseline("ours", cfg, ext, s, 1.0).TRV for s in seeds])
    try:
        half = np.mean([run_baseline("ours", cfg, ext, s, 0.5).TRV for s in seeds])
    except CapacityInfeasibleError:
        return float(full), None
    return float(full), float(half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    args = ap.parse_args()
    seeds = parse_seed_spec(args.seeds)
    base_dir = Path(__file__).parent.parent

    print("=" * 76)
    print("ERROR (OVERESTIMATION) RATE  "
          f"(sigma>=tau_h & phi>{HIGH_PHI_THRESHOLD} & "
          f"true_yield<{BAD_YIELD_THRESHOLD})")
    print("=" * 76)
    print(f"{'Scenario':<24} {'extr.':<8} {'bad':>5} {'score-skip':>10} {'total':>7} "
          f"{'cond.%':>8} {'prev.%':>8}")
    print("-" * 76)
    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        cache = _load_llm_cache(base_dir, key)
        rows = [("keyword", error_rate(cfg, key, seeds))]
        if cache:
            rows.append(("LLM", error_rate(cfg, key, seeds, llm_cache=cache)))
        for j, (name, e) in enumerate(rows):
            tag = label if j == 0 else ""
            note = f"  (miss={e['miss']})" if e.get("miss") else ""
            conditional = (
                f"{100.0 * e['rate_conditional']:.2f}%"
                if np.isfinite(e["rate_conditional"]) else "n/a"
            )
            prevalence = f"{100.0 * e['rate_prevalence']:.2f}%"
            print(f"{tag:<24} {name:<8} {e['bad']:>5} "
                  f"{e['high_score_skip']:>10} {e['tot']:>7} "
                  f"{conditional:>8} {prevalence:>8}{note}")
    print("-" * 76)
    print("cond.% = bad / high-condition, high-signal-quality skips")
    print("prev.% = bad / evaluated assets (batch prevalence)")
    print("LLM rows read phi/sigma from the cached DeepSeek responses (same asset stream).")

    print()
    print("=" * 68)
    print("CAPACITY SENSITIVITY  (SSADS-Keyword TRV: half capacity vs full)")
    print("=" * 68)
    print(f"{'Scenario':<28} {'full TRV':>14} {'half TRV':>14} {'change':>9}")
    print("-" * 68)
    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        full, half = capacity_sensitivity(cfg, key, seeds)
        if half is None:
            print(f"{label:<28} {full:>14,.0f} {'infeasible':>14} {'n/a':>9}")
        else:
            chg = 100.0 * (half - full) / full if full else float("nan")
            print(f"{label:<28} {full:>14,.0f} {half:>14,.0f} {chg:>8.1f}%")

    print()
    print("=" * 68)
    print("CASE STUDIES  (S1 deterministic keyword extractor)")
    print("=" * 68)
    print(f"{'Case':<28} {'phi':>8} {'sigma':>8}")
    print("-" * 68)
    ext = KeywordExtractor("s1")
    for label, note in CASE_NOTES:
        result = ext.extract(note, None)
        print(f"{label:<28} {result.phi:>8.3f} {result.sigma:>8.3f}")


if __name__ == "__main__":
    main()
