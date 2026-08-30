"""Ablation study: isolates where the framework's value comes from.

Addresses the reviewer question: are SSADS's inspection savings just an artifact
of inspecting fewer assets, or does signal-quality-guided *targeting* actually
allocate the inspection budget better?

For each scenario we compare, over the same 30 seeds (paired):

  SSADS-Keyword     : adaptive inspection by signal quality + greedy optimizer.
  Matched-cost random: assign the SAME counts of skip/quick/full decisions as
                       SSADS-Keyword, but randomly across assets.
  No-adaptive       : semantic phi + inspect EVERY asset (no skipping) + optimizer.
  Blind recovery    : no condition signal, no inspection + optimizer.

The key contrast is SSADS-Keyword vs matched-cost random: equal inspection budget,
so any TRV gap is attributable to signal-quality-guided targeting, not to inspecting less.

Usage:
    python scripts/run_ablation.py --seeds 0-29
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.s2s.extractors.base import NullExtractor
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.pipeline import load_config, run_pipeline
from src.s2s.randomness import parse_seed_spec

SCENARIOS = [
    ("configs/s1_it.yaml", "s1", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "s2", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "s3", "S3: Consumer Electronics"),
]


def run_scenario(cfg, key, seeds):
    ext = KeywordExtractor(key)
    rows: dict[str, dict[str, list[float]]] = {
        method: {"TRV": [], "ICS": [], "finsp": []}
        for method in ["ours", "matched_random", "no_adaptive", "blind"]
    }

    for s in seeds:
        ours = run_pipeline(cfg, ext, s, inspection_mode="policy")
        counts = {
            0: ours.inspection_skip,
            1: ours.inspection_quick,
            2: ours.inspection_full,
        }

        matched = run_pipeline(
            cfg,
            ext,
            s,
            inspection_mode="random_levels",
            random_level_counts=counts,
        )

        # No-adaptive: semantic phi but inspect everything.
        noad = run_pipeline(cfg, ext, s, inspection_mode="full")

        blind = run_pipeline(cfg, NullExtractor(), s, inspection_mode="none")

        for m, r in [("ours", ours), ("matched_random", matched),
                     ("no_adaptive", noad), ("blind", blind)]:
            rows[m]["TRV"].append(r.TRV)
            rows[m]["ICS"].append(r.ICS)
            rows[m]["finsp"].append(r.frac_inspected)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    args = ap.parse_args()
    seeds = parse_seed_spec(args.seeds)
    base = Path(__file__).parent.parent

    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base / cfg_path)
        rows = run_scenario(cfg, key, seeds)
        print("=" * 74)
        print(f"{label}   (mean over {len(seeds)} seeds)")
        print("=" * 74)
        print(f"{'Variant':<18}{'TRV ($K)':>12}{'ICS ($K)':>12}{'%inspected':>13}")
        print("-" * 74)
        for m, name in [("ours", "SSADS-Keyword"),
                        ("matched_random", "Matched-cost random"),
                        ("no_adaptive", "No-adaptive (all)"),
                        ("blind", "Blind recovery")]:
            trv = np.mean(rows[m]["TRV"]) / 1000
            ics = np.mean(rows[m]["ICS"]) / 1000
            fin = 100 * np.mean(rows[m]["finsp"])
            print(f"{name:<18}{trv:>12,.0f}{ics:>12,.0f}{fin:>12.0f}%")
        # Paired test: identical inspection depth counts, different targeting.
        a = np.array(rows["ours"]["TRV"]); b = np.array(rows["matched_random"]["TRV"])
        d = a.mean() - b.mean()
        print("-" * 74)
        if not np.allclose(a, b):
            _w, p = stats.wilcoxon(a, b)
            print(f"SSADS-Keyword vs matched-cost random: "
                  f"{d:+,.0f} dollars TRV, paired Wilcoxon p={p:.2e}")
        else:
            print("SSADS-Keyword == matched-cost random (identical)")
        print("NOTE: equal inspection spend, so the gap measures targeting value,"
              " not inspecting-less. p reflects seed stability only.")
        print()


if __name__ == "__main__":
    main()
