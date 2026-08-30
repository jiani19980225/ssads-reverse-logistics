"""Reviewer-requested experiment bundle.

This script adds the missing audit tables requested by the AIBThings
reviewers without changing the core benchmark protocol:

1. Inspection-depth counts and percentages for every baseline/scenario.
2. Per-asset TRV and seed-level TRV variance.
3. Threshold sensitivity around the submitted tau_l/tau_h settings.
4. Note-noise sensitivity for omission and severity-mislabel rates.
5. Sigma selective-risk summaries (sigma is informativeness, not probability).
   This includes disposition and realized-value outcomes for bad high-score skips.
6. Greedy-ranking ablation and an exact MILP optimality-gap audit.
7. Held-out vocabulary families, emulating an unwritten condition family.
8. Data-profile tables for note length, vocabulary diversity, and condition mix.
9. Scenario-parameter and LLM-cache coverage audits.

The LLM rows are cache-only. A missing cache entry is an error rather than a
reason to call an external API during a reproducibility audit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generators.common import generate_assets
from src.s2s.baselines.runner import (
    CombinedExtractor,
    RandomExtractor,
    StructuredOnlyExtractor,
    train_combined_model,
    train_structured_model,
)
from src.s2s.decision_engine import (
    exact_expected_objective,
    greedy_expected_objective,
    optimality_gap_percent,
    remaining_processing_capacity,
)
from src.s2s.extractors.base import NullExtractor
from src.s2s.extractors.deepseek import DeepSeekExtractor
from src.s2s.extractors.keyword import _SIGNALS, KeywordExtractor
from src.s2s.extractors.strong import StrongExtractor
from src.s2s.pipeline import load_config, run_pipeline
from src.s2s.randomness import parse_seed_spec, rng_for

SCENARIOS = [
    ("configs/s1_it.yaml", "s1", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "s2", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "s3", "S3: Consumer Electronics"),
]

METHODS = [
    "random", "rule_based", "structured", "combined", "oracle", "no_signal",
    "semantic_only", "ours",
]

THRESHOLD_VARIANTS = [
    ("default", None),
    ("more_skip", {"tau_h_delta": -0.10, "tau_l_delta": 0.00}),
    ("less_skip", {"tau_h_delta": 0.20, "tau_l_delta": 0.00}),
    ("less_full", {"tau_h_delta": 0.00, "tau_l_delta": -0.10}),
    ("more_full", {"tau_h_delta": 0.00, "tau_l_delta": 0.10}),
]

NOISE_VARIANTS = [
    ("default", None),
    ("no_noise", {"p_omit": 0.00, "p_mislabel": 0.00}),
    ("high_omit", {"p_omit": 0.30, "p_mislabel": 0.25}),
    ("high_mislabel", {"p_omit": 0.15, "p_mislabel": 0.40}),
    ("high_both", {"p_omit": 0.30, "p_mislabel": 0.40}),
]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scenario_cfg(base_dir: Path, cfg_path: str) -> dict:
    return load_config(base_dir / cfg_path)


def llm_cache(base_dir: Path, key: str) -> dict:
    p = base_dir / "outputs" / "llm_cache" / f"{key}_deepseek_deepseek-chat.json"
    return {
        k: tuple(v)
        for k, v in json.loads(p.read_text(encoding="utf-8")).items()
    } if p.exists() else {}


def semantic_extractor(kind: str, key: str, base_dir: Path):
    if kind == "keyword":
        return KeywordExtractor(key)
    if kind == "strong":
        return StrongExtractor(key)
    if kind == "llm":
        return DeepSeekExtractor(
            key,
            response_cache=llm_cache(base_dir, key),
            allow_live=False,
        )
    raise ValueError(f"Unknown extractor kind: {kind}")


def method_spec(method: str, cfg: dict, key: str, base_dir: Path, semantic_kind: str):
    """Return extractor, inspection mode, allocator, and ranking flag."""
    if method == "random":
        return RandomExtractor(), "none", "random", False
    if method == "rule_based":
        return NullExtractor(), "none", "fifo", False
    if method == "structured":
        return StructuredOnlyExtractor(train_structured_model(cfg)), "full", "greedy", False
    if method == "combined":
        return CombinedExtractor(train_combined_model(cfg), key), "policy", "greedy", False
    if method == "oracle":
        return NullExtractor(), "oracle", "greedy", False
    if method == "no_signal":
        return NullExtractor(), "none", "greedy", False
    if method == "semantic_only":
        return semantic_extractor(semantic_kind, key, base_dir), "policy", "threshold", False
    if method == "ours":
        return semantic_extractor(semantic_kind, key, base_dir), "policy", "greedy", False
    if method == "ours_no_ranking":
        return semantic_extractor(semantic_kind, key, base_dir), "policy", "greedy", True
    raise ValueError(f"Unknown method: {method}")


def apply_threshold_variant(cfg: dict, variant: dict | None) -> dict:
    out = copy.deepcopy(cfg)
    if not variant:
        return out
    tau_h = out["thresholds"]["tau_h"] + variant["tau_h_delta"]
    tau_l = out["thresholds"]["tau_l"] + variant["tau_l_delta"]
    tau_h = float(np.clip(tau_h, 0.05, 0.95))
    tau_l = float(np.clip(tau_l, 0.00, tau_h - 0.01))
    out["thresholds"] = {"tau_h": tau_h, "tau_l": tau_l}
    return out


def apply_noise_variant(cfg: dict, variant: dict | None) -> dict:
    out = copy.deepcopy(cfg)
    if variant:
        out["note_noise"] = dict(variant)
    return out


def _generate_paired_noise_assets(
    config: dict,
    generation_rng: np.random.Generator,
    *,
    seed: int,
) -> list[dict]:
    return generate_assets(
        config,
        generation_rng,
        rng_for(seed, "note_sensitivity"),
    )


def run_debug(
    cfg: dict,
    key: str,
    base_dir: Path,
    method: str,
    seed: int,
    semantic_kind: str = "keyword",
    capacity_fraction: float = 1.0,
    asset_generator=None,
) -> dict:
    """Run the canonical pipeline and retain per-asset audit values."""
    extractor, inspection_mode, allocator, disable_ranking = method_spec(
        method, cfg, key, base_dir, semantic_kind
    )
    metrics, assets, results = run_pipeline(
        cfg,
        extractor,
        seed,
        capacity_fraction=capacity_fraction,
        disable_greedy_ranking=disable_ranking,
        allocator=allocator,
        inspection_mode=inspection_mode,
        asset_generator=asset_generator,
        return_details=True,
    )
    per_asset_net = np.array([r["realized_value"] - r["inspection_cost"] for r in results])
    return {
        "metrics": metrics,
        "policy_skip": metrics.inspection_skip,
        "policy_quick": metrics.inspection_quick,
        "policy_full": metrics.inspection_full,
        "per_asset_mean": float(per_asset_net.mean()),
        "per_asset_std": float(per_asset_net.std(ddof=1)) if len(per_asset_net) > 1 else 0.0,
        "assets": assets,
        "results": results,
    }


def summarize_runs(rows: list[dict], n_assets: int) -> dict:
    trv_mean, trv_std = mean_std([r["metrics"].TRV for r in rows])
    rpr_mean, rpr_std = mean_std([r["metrics"].RPR for r in rows])
    ics_mean, ics_std = mean_std([r["metrics"].ICS for r in rows])
    inspected_mean, inspected_std = mean_std([r["metrics"].frac_inspected for r in rows])
    pa_mean, pa_std_across_seeds = mean_std([r["per_asset_mean"] for r in rows])
    within_pa_std_mean, _ = mean_std([r["per_asset_std"] for r in rows])
    skip_mean, skip_std = mean_std([r["policy_skip"] for r in rows])
    quick_mean, quick_std = mean_std([r["policy_quick"] for r in rows])
    full_mean, full_std = mean_std([r["policy_full"] for r in rows])
    return {
        "n_assets": n_assets,
        "TRV_mean": round(trv_mean, 3),
        "TRV_std": round(trv_std, 3),
        "per_asset_TRV_mean": round(pa_mean, 3),
        "per_asset_TRV_seed_std": round(pa_std_across_seeds, 3),
        "per_asset_within_batch_std_mean": round(within_pa_std_mean, 3),
        "RPR_mean": round(rpr_mean, 6),
        "RPR_std": round(rpr_std, 6),
        "ICS_mean": round(ics_mean, 3),
        "ICS_std": round(ics_std, 3),
        "policy_skip_mean": round(skip_mean, 3),
        "policy_skip_std": round(skip_std, 3),
        "policy_skip_pct": round(skip_mean / n_assets, 6),
        "policy_quick_mean": round(quick_mean, 3),
        "policy_quick_std": round(quick_std, 3),
        "policy_quick_pct": round(quick_mean / n_assets, 6),
        "policy_full_mean": round(full_mean, 3),
        "policy_full_std": round(full_std, 3),
        "policy_full_pct": round(full_mean / n_assets, 6),
        "charged_inspected_pct_mean": round(inspected_mean, 6),
        "charged_inspected_pct_std": round(inspected_std, 6),
    }


def run_main_tables(base_dir: Path, seeds: list[int]) -> tuple[list[dict], list[dict]]:
    inspection_rows, value_rows = [], []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        for method in METHODS:
            runs = [run_debug(cfg, key, base_dir, method, seed) for seed in seeds]
            summary = summarize_runs(runs, cfg["n_assets"])
            base = {
                "scenario": label,
                "method": method,
                "semantic_extractor": (
                    "keyword+structured" if method == "combined"
                    else "keyword" if method in {"semantic_only", "ours"}
                    else "n/a"
                ),
                "n_seeds": len(seeds),
            }
            inspection_rows.append({
                **base,
                "n_assets": summary["n_assets"],
                "policy_skip_mean": summary["policy_skip_mean"],
                "policy_skip_std": summary["policy_skip_std"],
                "policy_skip_pct": summary["policy_skip_pct"],
                "policy_quick_mean": summary["policy_quick_mean"],
                "policy_quick_std": summary["policy_quick_std"],
                "policy_quick_pct": summary["policy_quick_pct"],
                "policy_full_mean": summary["policy_full_mean"],
                "policy_full_std": summary["policy_full_std"],
                "policy_full_pct": summary["policy_full_pct"],
                "charged_inspected_pct_mean": summary["charged_inspected_pct_mean"],
                "charged_inspected_pct_std": summary["charged_inspected_pct_std"],
            })
            value_rows.append({
                **base,
                "TRV_mean": summary["TRV_mean"],
                "TRV_std": summary["TRV_std"],
                "per_asset_TRV_mean": summary["per_asset_TRV_mean"],
                "per_asset_TRV_seed_std": summary["per_asset_TRV_seed_std"],
                "per_asset_within_batch_std_mean": summary["per_asset_within_batch_std_mean"],
                "RPR_mean": summary["RPR_mean"],
                "RPR_std": summary["RPR_std"],
                "ICS_mean": summary["ICS_mean"],
                "ICS_std": summary["ICS_std"],
            })
    return inspection_rows, value_rows


def run_feature_signal_quality(base_dir: Path, seeds: list[int]) -> list[dict]:
    """Compare pre-inspection text-only, structured-only, and combined signals."""
    rows = []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        extractors = {
            "text_only_keyword": KeywordExtractor(key),
            "structured_only": StructuredOnlyExtractor(train_structured_model(cfg)),
            "combined_keyword_structured": CombinedExtractor(train_combined_model(cfg), key),
        }
        for method, extractor in extractors.items():
            predicted, target = [], []
            for seed in seeds:
                assets = generate_assets(cfg, rng_for(seed, "generation"))
                extract_rng = rng_for(seed, "extraction")
                for asset in assets:
                    result = extractor.extract(asset["text"], extract_rng, asset=asset)
                    predicted.append(result.phi)
                    target.append(asset["true_yield_factor"])
            predicted_arr = np.asarray(predicted, dtype=float)
            target_arr = np.asarray(target, dtype=float)
            correlation = (
                float(stats.pearsonr(predicted_arr, target_arr).statistic)
                if np.std(predicted_arr) > 1e-12 else float("nan")
            )
            rows.append({
                "scenario": label,
                "method": method,
                "n_seeds": len(seeds),
                "n_records": len(predicted),
                "pearson_r_phi_vs_latent_factor": round(correlation, 6),
                "mean_absolute_error": round(
                    float(np.mean(np.abs(predicted_arr - target_arr))), 6
                ),
                "training_population": (
                    "none" if method == "text_only_keyword"
                    else "4000 independent assets, seed 99999"
                ),
            })
    return rows


def run_threshold_sensitivity(base_dir: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for cfg_path, key, label in SCENARIOS:
        base_cfg = scenario_cfg(base_dir, cfg_path)
        for variant_name, variant in THRESHOLD_VARIANTS:
            cfg = apply_threshold_variant(base_cfg, variant)
            runs = [run_debug(cfg, key, base_dir, "ours", seed) for seed in seeds]
            s = summarize_runs(runs, cfg["n_assets"])
            rows.append({
                "scenario": label,
                "variant": variant_name,
                "n_seeds": len(seeds),
                "tau_l": cfg["thresholds"]["tau_l"],
                "tau_h": cfg["thresholds"]["tau_h"],
                "TRV_mean": s["TRV_mean"],
                "TRV_std": s["TRV_std"],
                "RPR_mean": s["RPR_mean"],
                "ICS_mean": s["ICS_mean"],
                "policy_skip_pct": s["policy_skip_pct"],
                "policy_quick_pct": s["policy_quick_pct"],
                "policy_full_pct": s["policy_full_pct"],
                "charged_inspected_pct_mean": s["charged_inspected_pct_mean"],
            })
    return rows


def run_noise_sensitivity(base_dir: Path, seeds: list[int]) -> list[dict]:
    """Vary note corruption while holding each seed's latent assets fixed."""
    rows = []
    for cfg_path, key, label in SCENARIOS:
        base_cfg = scenario_cfg(base_dir, cfg_path)
        for variant_name, variant in NOISE_VARIANTS:
            cfg = apply_noise_variant(base_cfg, variant)
            runs = [
                run_debug(
                    cfg,
                    key,
                    base_dir,
                    "ours",
                    seed,
                    asset_generator=partial(_generate_paired_noise_assets, seed=seed),
                )
                for seed in seeds
            ]
            s = summarize_runs(runs, cfg["n_assets"])
            noise = cfg.get("note_noise", {"p_omit": 0.15, "p_mislabel": 0.25})
            rows.append({
                "scenario": label,
                "variant": variant_name,
                "n_seeds": len(seeds),
                "p_omit": noise.get("p_omit", 0.15),
                "p_mislabel": noise.get("p_mislabel", 0.25),
                "population_protocol": "shared_latent_separate_note_rng",
                "TRV_mean": s["TRV_mean"],
                "TRV_std": s["TRV_std"],
                "RPR_mean": s["RPR_mean"],
                "ICS_mean": s["ICS_mean"],
                "policy_skip_pct": s["policy_skip_pct"],
                "policy_quick_pct": s["policy_quick_pct"],
                "policy_full_pct": s["policy_full_pct"],
            })
    return rows


def run_greedy_ablation(base_dir: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        for method in ["ours_no_ranking", "ours"]:
            runs = [run_debug(cfg, key, base_dir, method, seed) for seed in seeds]
            s = summarize_runs(runs, cfg["n_assets"])
            rows.append({
                "scenario": label,
                "method": method,
                "n_seeds": len(seeds),
                "TRV_mean": s["TRV_mean"],
                "TRV_std": s["TRV_std"],
                "per_asset_TRV_mean": s["per_asset_TRV_mean"],
                "RPR_mean": s["RPR_mean"],
                "ICS_mean": s["ICS_mean"],
                "policy_skip_pct": s["policy_skip_pct"],
                "policy_quick_pct": s["policy_quick_pct"],
                "policy_full_pct": s["policy_full_pct"],
            })
    return rows


def run_exact_optimality_gap(base_dir: Path, seeds: list[int]) -> list[dict]:
    """Compare the value-density heuristic with exact 0-1 allocation."""
    rows = []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        greedy_values, exact_values, gaps = [], [], []
        for seed in seeds:
            run = run_debug(cfg, key, base_dir, "ours", seed)
            assets = run["assets"]
            cap_key = "weekly_hours" if "weekly_hours" in cfg["capacity"] else "daily_hours"
            processing_capacity = remaining_processing_capacity(
                assets, cfg["capacity"][cap_key] * 60
            )
            greedy_value, _ = greedy_expected_objective(assets, processing_capacity)
            exact_value = exact_expected_objective(assets, processing_capacity)
            gap = optimality_gap_percent(greedy_value, exact_value)
            greedy_values.append(greedy_value)
            exact_values.append(exact_value)
            gaps.append(gap)
        rows.append({
            "scenario": label,
            "n_seeds": len(seeds),
            "n_assets_per_batch": cfg["n_assets"],
            "greedy_expected_margin_mean": round(float(np.mean(greedy_values)), 3),
            "exact_expected_margin_mean": round(float(np.mean(exact_values)), 3),
            "optimality_gap_pct_mean": round(float(np.mean(gaps)), 6),
            "optimality_gap_pct_max": round(float(np.max(gaps)), 6),
        })
    return rows


def run_matched_cost_ablation(base_dir: Path, seeds: list[int]) -> list[dict]:
    """Randomize inspection placement while preserving all depth counts."""
    rows = []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        ours_values, matched_values = [], []
        ours_rpr, matched_rpr = [], []
        for seed in seeds:
            extractor = KeywordExtractor(key)
            ours = run_pipeline(cfg, extractor, seed, inspection_mode="policy")
            counts = {
                0: ours.inspection_skip,
                1: ours.inspection_quick,
                2: ours.inspection_full,
            }
            matched = run_pipeline(
                cfg,
                KeywordExtractor(key),
                seed,
                inspection_mode="random_levels",
                random_level_counts=counts,
            )
            ours_values.append(ours.TRV)
            matched_values.append(matched.TRV)
            ours_rpr.append(ours.RPR)
            matched_rpr.append(matched.RPR)
        ours_arr = np.asarray(ours_values)
        matched_arr = np.asarray(matched_values)
        if np.allclose(ours_arr, matched_arr):
            p_value: float | str = "n/a"
        else:
            p_value = float(stats.wilcoxon(ours_arr, matched_arr).pvalue)
        rows.append({
            "scenario": label,
            "n_seeds": len(seeds),
            "ours_TRV_mean": round(float(ours_arr.mean()), 3),
            "matched_random_TRV_mean": round(float(matched_arr.mean()), 3),
            "ours_minus_matched_TRV": round(float((ours_arr - matched_arr).mean()), 3),
            "paired_wilcoxon_p": p_value,
            "ours_RPR_mean": round(float(np.mean(ours_rpr)), 6),
            "matched_random_RPR_mean": round(float(np.mean(matched_rpr)), 6),
            "inspection_cost_identical": True,
        })
    return rows


def sigma_records(base_dir: Path, cfg: dict, key: str, extractor_kind: str, seeds: list[int]) -> list[dict]:
    ext = semantic_extractor(extractor_kind, key, base_dir)
    rows = []
    for seed in seeds:
        gen_rng = rng_for(seed, "generation")
        extract_rng = rng_for(seed, "extraction")
        for a in generate_assets(cfg, gen_rng):
            res = ext.extract(a["text"], extract_rng, asset=a)
            rows.append({
                "phi": res.phi,
                "sigma": res.sigma,
                "true_yield": a["true_yield_factor"],
                "used_fallback": bool(getattr(res, "used_fallback", False)),
            })
    return rows


def selective_risk_summary(records: list[dict], tau_h: float, tolerance: float) -> tuple[dict, list[dict]]:
    if not records:
        return {}, []
    sig = np.array([r["sigma"] for r in records], dtype=float)
    phi = np.array([r["phi"] for r in records], dtype=float)
    y = np.array([r["true_yield"] for r in records], dtype=float)
    absolute_error = np.abs(phi - y)
    accurate = (absolute_error <= tolerance).astype(float)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
    bin_rows = []
    for lo, hi in bins:
        mask = (sig >= lo) & (sig < hi)
        if not np.any(mask):
            row = {
                "sigma_bin": (
                    f"[{lo:.1f},{min(hi, 1.0):.1f}]" if hi > 1.0
                    else f"[{lo:.1f},{hi:.1f})"
                ),
                "n": 0,
                "mean_sigma": "n/a",
                "mean_abs_phi_error": "n/a",
                "within_tolerance_rate": "n/a",
            }
            bin_rows.append(row)
            continue
        n = int(mask.sum())
        mean_sigma = float(sig[mask].mean())
        acc = float(accurate[mask].mean())
        bin_rows.append({
            "sigma_bin": (
                f"[{lo:.1f},{min(hi, 1.0):.1f}]" if hi > 1.0
                else f"[{lo:.1f},{hi:.1f})"
            ),
            "n": n,
            "mean_sigma": round(mean_sigma, 6),
            "mean_abs_phi_error": round(float(absolute_error[mask].mean()), 6),
            "within_tolerance_rate": round(acc, 6),
        })
    high_score_skip = (sig >= tau_h) & (phi > 0.7)
    bad_high_score_skip = high_score_skip & (y < 0.30)
    if np.std(sig) > 1e-12 and np.std(absolute_error) > 1e-12:
        sigma_error_correlation: float | str = round(
            float(np.corrcoef(sig, absolute_error)[0, 1]), 6
        )
    else:
        sigma_error_correlation = "n/a"
    summary = {
        "n": len(records),
        "tolerance_abs_phi_error": tolerance,
        "mean_abs_phi_error": round(float(absolute_error.mean()), 6),
        "within_tolerance_rate": round(float(accurate.mean()), 6),
        "sigma_abs_error_correlation": sigma_error_correlation,
        "high_score_skip_count": int(high_score_skip.sum()),
        "high_score_skip_pct": round(float(high_score_skip.mean()), 6),
        "bad_high_score_skip_count": int(bad_high_score_skip.sum()),
        "bad_high_score_skip_cond_rate": round(
            float(bad_high_score_skip.sum() / high_score_skip.sum()), 6
        ) if high_score_skip.sum() else "n/a",
    }
    return summary, bin_rows


def run_sigma_selective_risk(base_dir: Path, seeds: list[int], tolerance: float, include_llm: bool) -> tuple[list[dict], list[dict]]:
    summary_rows, bin_rows_all = [], []
    kinds = ["keyword", "strong"] + (["llm"] if include_llm else [])
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        for kind in kinds:
            records = sigma_records(base_dir, cfg, key, kind, seeds)
            summary, bin_rows = selective_risk_summary(records, cfg["thresholds"]["tau_h"], tolerance)
            summary_rows.append({
                "scenario": label,
                "extractor": kind,
                "n_seeds": len(seeds),
                **summary,
            })
            for b in bin_rows:
                bin_rows_all.append({
                    "scenario": label,
                    "extractor": kind,
                    "n_seeds": len(seeds),
                    **b,
                })
    return summary_rows, bin_rows_all


def run_bad_skip_outcomes(
    base_dir: Path, seeds: list[int], include_llm: bool
) -> list[dict]:
    """Link risky high-score skips to disposition and realized economic outcome."""
    rows = []
    kinds = ["keyword", "strong"] + (["llm"] if include_llm else [])
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        tau_h = cfg["thresholds"]["tau_h"]
        for kind in kinds:
            bad = []
            for seed in seeds:
                run = run_debug(cfg, key, base_dir, "ours", seed, kind)
                for asset, result in zip(
                    run["assets"], run["results"], strict=True
                ):
                    is_bad_skip = (
                        asset["inspection"].level == 0
                        and asset["sigma"] >= tau_h
                        and asset["phi"] > 0.7
                        and asset["true_yield_factor"] < 0.30
                    )
                    if is_bad_skip:
                        bad.append({
                            "disposition": result["disposition"],
                            "net_value": result["realized_value"] - result["inspection_cost"],
                        })
            values = np.array([item["net_value"] for item in bad], dtype=float)
            counts = {
                disposition: sum(item["disposition"] == disposition for item in bad)
                for disposition in ("refurbish", "component_recovery", "scrap")
            }
            rows.append({
                "scenario": label,
                "extractor": kind,
                "n_seeds": len(seeds),
                "bad_high_score_skip_count": len(bad),
                "routed_refurbish_count": counts["refurbish"],
                "routed_component_recovery_count": counts["component_recovery"],
                "routed_scrap_count": counts["scrap"],
                "realized_net_value_total": round(float(values.sum()), 3) if len(values) else 0.0,
                "realized_net_value_mean": round(float(values.mean()), 3)
                if len(values) else "n/a",
                "negative_net_value_count": int((values < 0).sum()) if len(values) else 0,
                "negative_net_value_rate": round(float((values < 0).mean()), 6)
                if len(values) else "n/a",
            })
    return rows


def run_held_out_vocabulary(base_dir: Path, seeds: list[int]) -> list[dict]:
    """Blank one keyword family at a time, thresholds frozen, and re-evaluate.

    Emulates a condition family the dictionary was never written for: its notes
    match nothing and fall back to the uninformative prior. Reports signal
    quality, fallback share, bad-skip rate, and the TRV consequence.
    """
    rows = []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        tau_h = cfg["thresholds"]["tau_h"]
        families = ("none",) + tuple(sorted(_SIGNALS[key]))
        baseline_trv = None
        for family in families:
            dropped = () if family == "none" else (family,)
            extractor = KeywordExtractor(key, drop_families=dropped)
            phi_all, y_all, fallback = [], [], []
            bad = high_score_skip = 0
            for seed in seeds:
                for asset in generate_assets(cfg, rng_for(seed, "generation")):
                    result = extractor.extract(asset["text"], None)
                    phi_all.append(result.phi)
                    y_all.append(asset["true_yield_factor"])
                    fallback.append(result.used_fallback)
                    if result.sigma >= tau_h and result.phi > 0.7:
                        high_score_skip += 1
                        if asset["true_yield_factor"] < 0.30:
                            bad += 1
            trv = float(np.mean([
                run_pipeline(
                    cfg, KeywordExtractor(key, drop_families=dropped), seed,
                    inspection_mode="policy",
                ).TRV
                for seed in seeds
            ]))
            if family == "none":
                baseline_trv = trv
            phi_arr = np.asarray(phi_all, dtype=float)
            correlation = (
                round(float(stats.pearsonr(phi_arr, np.asarray(y_all)).statistic), 6)
                if np.std(phi_arr) > 1e-12 else "n/a"
            )
            rows.append({
                "scenario": label,
                "held_out_family": family,
                "n_seeds": len(seeds),
                "n_records": len(phi_all),
                "pearson_r_phi_vs_latent_factor": correlation,
                "fallback_rate": round(float(np.mean(fallback)), 6),
                "TRV_mean": round(trv, 3),
                "TRV_pct_change_vs_baseline": round(
                    100.0 * (trv - baseline_trv) / baseline_trv, 4
                ) if baseline_trv else 0.0,
                "high_score_skip_count": high_score_skip,
                "bad_high_score_skip_count": bad,
                "bad_high_score_skip_cond_rate": round(bad / high_score_skip, 6)
                if high_score_skip else "n/a",
            })
    return rows


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def percentile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else 0.0


def condition_rows(
    scenario: str,
    field: str,
    values: list[str],
) -> list[dict]:
    total = len(values)
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return [
        {
            "scenario": scenario,
            "condition_field": field,
            "condition": cond,
            "count": count,
            "pct": round(count / total, 6) if total else 0.0,
        }
        for cond, count in sorted(counts.items())
    ]


def run_data_profile(base_dir: Path, seeds: list[int]) -> tuple[list[dict], list[dict]]:
    profile_rows, condition_mix_rows = [], []
    for cfg_path, _key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        records = []
        for seed in seeds:
            gen_rng = rng_for(seed, "generation")
            records.extend(generate_assets(cfg, gen_rng))

        notes = [a["text"] for a in records]
        unique_notes = sorted(set(notes))
        word_counts = [len(tokenize(n)) for n in notes]
        unique_word_counts = [len(tokenize(n)) for n in unique_notes]
        all_tokens = [tok for n in notes for tok in tokenize(n)]
        unique_tokens = sorted(set(all_tokens))
        true_conditions = [a.get("true_condition", "") for a in records]
        observed_conditions = [a.get("observed_condition", "") for a in records]
        profile_rows.append({
            "scenario": label,
            "n_seeds": len(seeds),
            "n_records": len(records),
            "unique_notes": len(unique_notes),
            "unique_note_ratio": round(len(unique_notes) / len(records), 6) if records else 0.0,
            "mean_note_words": round(float(np.mean(word_counts)), 3),
            "median_note_words": round(float(np.median(word_counts)), 3),
            "p10_note_words": round(percentile(word_counts, 10), 3),
            "p90_note_words": round(percentile(word_counts, 90), 3),
            "mean_unique_note_words": round(float(np.mean(unique_word_counts)), 3),
            "vocab_size": len(unique_tokens),
            "type_token_ratio": round(len(unique_tokens) / len(all_tokens), 6) if all_tokens else 0.0,
            "true_condition_classes": len(set(true_conditions)),
            "observed_condition_classes": len(set(observed_conditions)),
        })
        condition_mix_rows.extend(condition_rows(label, "true_condition", true_conditions))
        condition_mix_rows.extend(condition_rows(label, "observed_condition", observed_conditions))
    return profile_rows, condition_mix_rows


def run_parameter_summary(base_dir: Path) -> list[dict]:
    rows = []
    for cfg_path, _key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        capacity_key = "weekly_hours" if "weekly_hours" in cfg["capacity"] else "daily_hours"
        yield_bounds = list(cfg["condition_yield_ranges"].values())
        rows.append({
            "scenario": label,
            "n_assets_per_batch": cfg["n_assets"],
            "asset_type_count": len(cfg["asset_types"]),
            "component_count": len(cfg["prices"]),
            "price_min": min(cfg["prices"].values()),
            "price_max": max(cfg["prices"].values()),
            "base_yield_min": min(cfg["base_yields"].values()),
            "base_yield_max": max(cfg["base_yields"].values()),
            "latent_yield_factor_min": min(bounds[0] for bounds in yield_bounds),
            "latent_yield_factor_max": max(bounds[1] for bounds in yield_bounds),
            "realized_yield_concentration": cfg["realized_yield_concentration"],
            "condition_distribution_json": json.dumps(
                cfg["note_distribution"], sort_keys=True
            ),
            "capacity_period": capacity_key.replace("_hours", ""),
            "capacity_hours": cfg["capacity"][capacity_key],
            "tau_l": cfg["thresholds"]["tau_l"],
            "tau_h": cfg["thresholds"]["tau_h"],
            "note_p_omit": cfg.get("note_noise", {}).get("p_omit", 0.15),
            "note_p_mislabel": cfg.get("note_noise", {}).get("p_mislabel", 0.25),
            "inspection_skip_cost": cfg["inspection_costs"]["skip"]["cost"],
            "inspection_l1_cost": cfg["inspection_costs"]["l1"]["cost"],
            "inspection_l2_cost": cfg["inspection_costs"]["l2"]["cost"],
            "inspection_skip_min": cfg["inspection_costs"]["skip"]["time_min"],
            "inspection_l1_min": cfg["inspection_costs"]["l1"]["time_min"],
            "inspection_l2_min": cfg["inspection_costs"]["l2"]["time_min"],
            "inspection_l1_observation_sd": cfg["inspection_observation_noise"]["l1_sd"],
            "inspection_l2_observation_sd": cfg["inspection_observation_noise"]["l2_sd"],
            "inspection_l1_update_weight": cfg["inspection_update_weights"]["l1"],
            "inspection_l2_update_weight": cfg["inspection_update_weights"]["l2"],
            "routing_partial_threshold": cfg["routing_thresholds"]["partial"],
            "routing_recover_threshold": cfg["routing_thresholds"]["recover"],
            "partial_recovery_fraction": cfg["processing_costs"]["l1"]["recovery_fraction"],
            "scrap_cost": cfg["processing_costs"]["scrap"]["cost"],
            "scrap_time_min": cfg["processing_costs"]["scrap"]["time_min"],
            "rework_trigger_yield_below": cfg["processing_costs"].get("rework", {}).get(
                "trigger_yield_below", "n/a"
            ),
            "processing_costs_json": json.dumps(cfg["processing_costs"], sort_keys=True),
        })
    return rows


def run_llm_cache_audit(base_dir: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for cfg_path, key, label in SCENARIOS:
        cfg = scenario_cfg(base_dir, cfg_path)
        cache_path = base_dir / "outputs" / "llm_cache" / f"{key}_deepseek_deepseek-chat.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        generated_notes: set[str] = set()
        for seed in seeds:
            gen_rng = rng_for(seed, "generation")
            generated_notes.update(a["text"] for a in generate_assets(cfg, gen_rng))

        missing = [n for n in generated_notes if n not in cache]
        valid = 0
        invalid_covered = 0
        for note in generated_notes:
            payload = cache.get(note)
            ok = (
                isinstance(payload, list)
                and len(payload) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in payload)
                and 0.0 < float(payload[0]) <= 1.0
                and 0.0 <= float(payload[1]) <= 1.0
            )
            if ok:
                valid += 1
            elif payload is not None:
                invalid_covered += 1
        rows.append({
            "scenario": label,
            "model": "deepseek-chat",
            "api_date": "not_recorded",
            "cache_file": str(cache_path.relative_to(base_dir)),
            "cache_total_entries": len(cache),
            "generated_unique_notes": len(generated_notes),
            "covered_generated_notes": len(generated_notes) - len(missing),
            "missing_generated_notes": len(missing),
            "valid_cached_generated_entries": valid,
            "invalid_or_unparseable_covered_entries": invalid_covered,
            "live_parse_failures": "not_recorded",
        })
    return rows


def write_markdown_summary(out_dir: Path, tables: dict[str, list[dict]], include_llm: bool) -> None:
    lines = [
        "# Reviewer Experiment Summary",
        "",
        "Generated by `experiments/scripts/run_reviewer_experiments.py`.",
        "",
        f"LLM rows included: `{include_llm}`. LLM audit rows are strictly cache-only.",
        "",
        "## Output Files",
        "",
    ]
    for name in tables:
        lines.append(f"- `{name}.csv`")
    lines += [
        "",
        "## Notes",
        "",
        "- `policy_skip/quick/full` counts are pre-allocation inspection-policy decisions.",
        "- Every first-stage inspection is charged before recovery allocation; policy and charged inspection counts therefore agree.",
        "- Sigma denotes note informativeness, not a calibrated probability. The selective-risk tables report error and bad-skip rates without ECE/Brier.",
        "- Threshold sensitivity uses the main benchmark's paired seed protocol.",
        "- Note-noise sensitivity holds latent assets fixed with a separate deterministic note RNG; only note corruption changes across variants.",
        "- `data_profile` and `condition_mix` report generated synthetic text diversity and class balance.",
        "- `llm_cache_audit` validates cache coverage for generated notes, but the original live API date and parse-failure count were not recorded.",
    ]
    out_dir.joinpath("SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out-dir", default="outputs/reviewer_experiments")
    ap.add_argument("--include-llm", action="store_true",
                    help="Add cache-backed LLM rows to sigma selective-risk analysis.")
    ap.add_argument("--sigma-tolerance", type=float, default=0.20,
                    help="Accuracy event for sigma audit: abs(phi - true_yield) <= tolerance.")
    args = ap.parse_args()

    seeds = parse_seed_spec(args.seeds)
    if not 0.0 <= args.sigma_tolerance <= 1.0:
        ap.error("--sigma-tolerance must be in [0, 1]")
    base_dir = Path(__file__).parent.parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reviewer experiments] seeds={args.seeds}; out={out_dir}")
    print("[1/12] main inspection/value tables")
    inspection_rows, value_rows = run_main_tables(base_dir, seeds)
    print("[2/12] text/structured/combined signal comparison")
    feature_signal_rows = run_feature_signal_quality(base_dir, seeds)
    print("[3/12] threshold sensitivity")
    threshold_rows = run_threshold_sensitivity(base_dir, seeds)
    print("[4/12] note-noise sensitivity")
    noise_rows = run_noise_sensitivity(base_dir, seeds)
    print("[5/12] greedy-ranking ablation")
    greedy_rows = run_greedy_ablation(base_dir, seeds)
    print("[6/12] exact optimization audit")
    exact_rows = run_exact_optimality_gap(base_dir, seeds)
    print("[7/12] matched-cost inspection ablation")
    matched_rows = run_matched_cost_ablation(base_dir, seeds)
    print("[8/12] sigma selective-risk analysis")
    sigma_rows, sigma_bin_rows = run_sigma_selective_risk(
        base_dir, seeds, args.sigma_tolerance, args.include_llm
    )
    print("[9/12] bad-skip outcome audit")
    bad_skip_rows = run_bad_skip_outcomes(base_dir, seeds, args.include_llm)
    print("[10/13] held-out vocabulary family")
    held_out_rows = run_held_out_vocabulary(base_dir, seeds)
    print("[11/13] data profile")
    data_profile_rows, condition_mix_rows = run_data_profile(base_dir, seeds)
    print("[12/13] scenario parameter summary")
    parameter_rows = run_parameter_summary(base_dir)
    print("[13/13] LLM cache audit")
    llm_cache_rows = run_llm_cache_audit(base_dir, seeds)

    tables = {
        "inspection_depth_by_method": inspection_rows,
        "per_asset_value_by_method": value_rows,
        "feature_signal_quality": feature_signal_rows,
        "threshold_sensitivity": threshold_rows,
        "note_noise_sensitivity": noise_rows,
        "greedy_ranking_ablation": greedy_rows,
        "exact_optimality_gap": exact_rows,
        "matched_cost_inspection": matched_rows,
        "sigma_selective_risk_summary": sigma_rows,
        "sigma_selective_risk_bins": sigma_bin_rows,
        "bad_skip_outcomes": bad_skip_rows,
        "held_out_vocabulary": held_out_rows,
        "data_profile": data_profile_rows,
        "condition_mix": condition_mix_rows,
        "scenario_parameter_summary": parameter_rows,
        "llm_cache_audit": llm_cache_rows,
    }
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)
    write_markdown_summary(out_dir, tables, args.include_llm)
    print("[done]")
    for name, rows in tables.items():
        print(f"  {name}.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
