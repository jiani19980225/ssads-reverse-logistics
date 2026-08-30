"""Main result summary with paired comparisons across seeds.

Produces, for each scenario:
  - mean and std of TRV, RPR, and ICS across all seeds for every method
  - a paired-bootstrap 95% CI for the mean TRV lift over structured inspection
  - the paired Wilcoxon statistic and p-value (SSADS vs structured TRV)
  - a clearly labeled note on what the p-value does and does not measure

This is the canonical generator for the paper's main table. Numbers are whatever
the code produces; nothing is targeted. The same seed is used for every method
(shared asset population), so the primary comparison is correctly paired.

Usage:
    python scripts/run_summary.py --seeds 0-29
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.baselines.runner import run_baseline
from src.s2s.extractors.base import AbstractExtractor
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.extractors.strong import StrongExtractor
from src.s2s.pipeline import load_config
from src.s2s.randomness import parse_seed_spec

METHODS = [
    "random", "rule_based", "structured", "combined", "oracle", "no_signal",
    "semantic_only", "ours",
]
SCENARIOS = [
    ("configs/s1_it.yaml", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "S3: Consumer Electronics"),
]

STABILITY_NOTE = (
    "p-value reflects simulation stability across seeds, "
    "not cross-dataset generalization"
)

METHOD_LABELS = {
    "random": "Random routing",
    "rule_based": "FIFO routing",
    "structured": "Structured + noisy full inspection",
    "combined": "Structured + semantic adaptive inspection",
    "oracle": "Oracle full inspection",
    "no_signal": "No-signal / no-inspection",
    "semantic_only": "Semantic-only / threshold routing",
}
EXTRACTOR_LABELS = {
    "keyword": "SSADS-Keyword",
    "strong": "SSADS-Phrase",
    "llm": "SSADS-DeepSeek",
}
SUMMARY_WIDTH = 100


def paired_lift_ci(ours: np.ndarray, comparator: np.ndarray,
                   n_boot: int = 10_000) -> tuple[float, float]:
    """Paired-bootstrap CI for percentage lift in the ratio of sample means."""
    if len(ours) == 0 or len(ours) != len(comparator):
        raise ValueError("Paired arrays must be non-empty and have equal length")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    rng = np.random.default_rng(12345)
    indices = rng.integers(0, len(ours), size=(n_boot, len(ours)))
    ours_means = ours[indices].mean(axis=1)
    comparator_means = comparator[indices].mean(axis=1)
    if np.any(np.isclose(comparator_means, 0.0)):
        raise ValueError("Percentage lift is undefined for a zero comparator mean")
    lifts = 100.0 * (ours_means - comparator_means) / comparator_means
    return tuple(np.percentile(lifts, [2.5, 97.5]))


def summarize_scenario(
    cfg_path: Path,
    label: str,
    seeds: list[int],
    extractor_kind: str = "keyword",
    allow_live_llm: bool = False,
):
    config = load_config(cfg_path)
    scenario_key = config["name"].split("_")[0]
    cache_path: Path | None = None
    cache: dict = {}
    initial_cache_size = 0
    if extractor_kind == "strong":
        ext: AbstractExtractor = StrongExtractor(scenario_key)
    elif extractor_kind == "llm":
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        cache_path = (cfg_path.parent.parent / "outputs" / "llm_cache"
                      / f"{scenario_key}_deepseek_deepseek-chat.json")
        if cache_path.exists():
            cache = {
                k: tuple(v)
                for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()
            }
        initial_cache_size = len(cache)
        ext = DeepSeekExtractor(
            scenario_key,
            response_cache=cache,
            allow_live=allow_live_llm,
        )
        mode = "live access allowed" if allow_live_llm else "cache-only"
        print(f"  [llm cache: {len(cache)} entries; {mode}]")
    else:
        ext = KeywordExtractor(scenario_key)

    # Collect per-seed metrics for every method (same seeds -> paired).
    data: dict[str, dict[str, list[float]]] = {
        method: {"TRV": [], "RPR": [], "ICS": []} for method in METHODS
    }
    for m in METHODS:
        for s in seeds:
            res = run_baseline(m, config, ext, s)
            data[m]["TRV"].append(res.TRV)
            data[m]["RPR"].append(res.RPR)
            data[m]["ICS"].append(res.ICS)

    if cache_path is not None and len(cache) != initial_cache_size:
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps({key: list(value) for key, value in cache.items()}),
            encoding="utf-8",
        )
        temp_path.replace(cache_path)

    print("=" * SUMMARY_WIDTH)
    print(f"{label}   (mean +/- std over {len(seeds)} seeds)")
    print("=" * SUMMARY_WIDTH)
    print(f"{'Method':<42} {'TRV mean':>13} {'TRV std':>11} {'RPR':>14} {'ICS mean':>12}")
    print("-" * SUMMARY_WIDTH)
    for m in METHODS:
        trv = np.array(data[m]["TRV"])
        rpr = np.array(data[m]["RPR"])
        ics = np.array(data[m]["ICS"])
        trv_std = trv.std(ddof=1) if len(trv) > 1 else 0.0
        rpr_std = rpr.std(ddof=1) if len(rpr) > 1 else 0.0
        name = EXTRACTOR_LABELS[extractor_kind] if m == "ours" else METHOD_LABELS[m]
        print(f"{name:<42} {trv.mean():>13,.0f} {trv_std:>11,.0f} "
              f"{rpr.mean():>7.1%}+/-{rpr_std:>4.1%} {ics.mean():>12,.0f}")

    # Paired Wilcoxon: selected SSADS extractor vs structured noisy full inspection.
    ours = np.array(data["ours"]["TRV"])
    structured = np.array(data["structured"]["TRV"])
    diff = ours - structured
    lift_abs = diff.mean()
    lift_pct = 100.0 * lift_abs / structured.mean() if structured.mean() else float("nan")
    ci_lo, ci_hi = paired_lift_ci(ours, structured)
    print("-" * SUMMARY_WIDTH)
    ssads_label = EXTRACTOR_LABELS[extractor_kind]
    print(f"Lift ({ssads_label} - Structured + noisy full inspection): "
          f"{lift_abs:,.0f}  ({lift_pct:+.1f}%)")
    print(f"Paired-bootstrap 95% CI for percentage lift: [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")
    if np.allclose(diff, 0):
        print("Wilcoxon: undefined (all paired differences are zero)")
    else:
        w, p = stats.wilcoxon(ours, structured)
        print(f"Paired Wilcoxon ({ssads_label} vs structured noisy full inspection, "
              f"n={len(seeds)}): "
              f"W={w:.1f}, p={p:.2e}")
    print(f"NOTE: {STABILITY_NOTE}.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--extractor", default="keyword", choices=["keyword", "strong", "llm"],
                    help="Extractor used for SSADS and semantic-only routing. "
                         "'strong' = full-vocabulary phrase matcher (stronger signal); "
                         "closes the signal-quality-to-value loop deterministically.")
    ap.add_argument(
        "--allow-live-llm",
        action="store_true",
        help="Allow DeepSeek API calls on cache misses (disabled by default).",
    )
    args = ap.parse_args()
    seeds = parse_seed_spec(args.seeds)
    base_dir = Path(__file__).parent.parent
    print(f"[extractor for semantic methods: {args.extractor}]\n")
    for cfg_path, label in SCENARIOS:
        try:
            summarize_scenario(
                base_dir / cfg_path,
                label,
                seeds,
                args.extractor,
                args.allow_live_llm,
            )
        except RuntimeError as exc:
            ap.error(str(exc))


if __name__ == "__main__":
    main()
