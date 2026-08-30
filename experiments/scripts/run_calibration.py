"""Signal-quality analysis against the synthetic latent yield factor.

The extractor sees only text generated from a corrupted observation of the
latent condition. It does not receive ``true_yield_factor`` at runtime. The
benchmark intentionally makes text informative about that latent factor; this
script measures how much of that simulated signal the extractor recovers.

Two correlations are reported per scenario:
  - r(phi, factor): context-factor agreement with the latent yield factor
  - r(yhat, yield): value-weighted component-yield prior vs its latent target

Pooled across the given seeds for a stable estimate; per-seed mean/std also shown.

Usage:
    python scripts/run_calibration.py --seeds 0-29
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generators.common import generate_assets
from src.s2s.extractors.keyword import KeywordExtractor
from src.s2s.extractors.strong import StrongExtractor
from src.s2s.pipeline import load_config
from src.s2s.randomness import parse_seed_spec, rng_for

SCENARIOS = [
    ("configs/s1_it.yaml", "s1", "S1: IT Infrastructure"),
    ("configs/s2_aviation.yaml", "s2", "S2: Aviation MRO"),
    ("configs/s3_consumer.yaml", "s3", "S3: Consumer Electronics"),
]


def measure_signal_quality(config: dict, scenario_key: str, seeds: list[int],
                           extractor=None) -> dict:
    """Generate assets, run an extractor, correlate phi vs ground truth.

    Mirrors how the pipeline derives its RNG streams so the text/yields scored
    here are the same ones the simulation uses. `extractor` defaults to the
    scenario KeywordExtractor.
    """
    ext = extractor if extractor is not None else KeywordExtractor(scenario_key)
    base_yields = config["base_yields"]

    phi_all, yhat_all, factor_true_all, yield_true_all = [], [], [], []
    fb_all = []
    per_seed_r = []

    for seed in seeds:
        # Match the canonical pipeline streams exactly.
        gen_rng = rng_for(seed, "generation")
        extract_rng = rng_for(seed, "extraction")

        assets = generate_assets(config, gen_rng)

        phi_s, yhat_s, factor_true_s, yield_true_s = [], [], [], []
        fb_s = []
        for a in assets:
            res = ext.extract(a["text"], extract_rng, asset=a)
            fb_s.append(bool(getattr(res, "used_fallback", res.phi >= 0.999)))
            value_weights = np.array([
                count * config["prices"][component]
                for component, count in a["components"].items()
            ], dtype=float)
            component_yields = np.array([
                base_yields[component] for component in a["components"]
            ], dtype=float)
            mean_by = float(np.average(component_yields, weights=value_weights))
            phi_s.append(res.phi)
            yhat_s.append(res.phi * mean_by)
            factor_true_s.append(a["true_yield_factor"])
            yield_true_s.append(a["true_yield_factor"] * mean_by)

        phi_all += phi_s
        yhat_all += yhat_s
        factor_true_all += factor_true_s
        yield_true_all += yield_true_s
        fb_all += fb_s
        # Guard per-seed r(phi, true) against zero variance (all phi == 1.0).
        if np.std(phi_s) > 1e-9 and np.std(factor_true_s) > 1e-9:
            per_seed_r.append(stats.pearsonr(phi_s, factor_true_s)[0])

    phi_arr = np.asarray(phi_all, dtype=float)
    yhat_arr = np.asarray(yhat_all, dtype=float)
    factor_true_arr = np.asarray(factor_true_all, dtype=float)
    yield_true_arr = np.asarray(yield_true_all, dtype=float)

    out = {
        "n": len(phi_arr),
        "frac_fallback": float(np.mean(fb_all)) if fb_all else 0.0,  # explicit used_fallback flag
    }
    if np.std(phi_arr) > 1e-9:
        out["r_phi"] = float(stats.pearsonr(phi_arr, factor_true_arr)[0])
        out["r_yhat"] = float(stats.pearsonr(yhat_arr, yield_true_arr)[0])
        out["r_phi_per_seed_mean"] = float(np.mean(per_seed_r)) if per_seed_r else float("nan")
        out["r_phi_per_seed_std"] = float(np.std(per_seed_r)) if per_seed_r else float("nan")
    else:
        out["r_phi"] = float("nan")  # extractor emitted constant phi -> undefined
        out["r_yhat"] = float("nan")
        out["r_phi_per_seed_mean"] = float("nan")
        out["r_phi_per_seed_std"] = float("nan")
    return out


def _make_llm_extractor(
    provider: str,
    scenario_key: str,
    model: str,
    cache: dict,
    allow_live: bool,
):
    """Build the LLM extractor, sharing a response cache."""
    if provider == "deepseek":
        from src.s2s.extractors.deepseek import DeepSeekExtractor
        return DeepSeekExtractor(
            scenario_key,
            model=model,
            response_cache=cache,
            allow_live=allow_live,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def measure_llm_signal_quality(config: dict, scenario_key: str, seeds: list,
                               sample: int, cache_path: Path, model: str,
                               provider: str, allow_live: bool = False) -> dict:
    """Measure LLM signal quality on a deterministic subsample of notes.

    Collects (note, true_yield) pairs across the seeds, subsamples `sample`
    records with a fixed RNG (so the same records are scored every run), runs the
    chosen LLM extractor, and correlates phi against true_yield. Cached notes are
    replayed offline. Cache misses raise unless live access is explicitly enabled.
    """
    records = []
    for seed in seeds:
        gen_rng = rng_for(seed, "generation")
        assets = generate_assets(config, gen_rng)
        for a in assets:
            records.append((a["text"], a["true_yield_factor"]))

    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        raise ValueError("sample must be a positive integer")
    pick = np.random.default_rng(12345).permutation(len(records))[:sample]
    sampled = [records[i] for i in pick]

    cache = {}
    if cache_path and cache_path.exists():
        cache = {
            k: tuple(v)
            for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()
        }

    ext = _make_llm_extractor(provider, scenario_key, model, cache, allow_live)
    phis, ys = [], []
    for text, y in sampled:
        res = ext.extract(text, None)
        phis.append(res.phi)
        ys.append(y)

    if cache_path and allow_live:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({k: list(v) for k, v in cache.items()}),
            encoding="utf-8",
        )

    r = stats.pearsonr(phis, ys)[0] if np.std(phis) > 1e-9 else float("nan")
    return {"n": len(phis), "r_phi": float(r)}


_DEFAULT_LLM_MODEL = {
    "deepseek": "deepseek-chat",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--llm", action="store_true",
                    help="Also run the cache-backed LLM extractor on a bounded sample.")
    ap.add_argument("--llm-provider", default="deepseek",
                    choices=["deepseek"],
                    help="LLM provider (DEEPSEEK_API_KEY). DeepSeek is the one "
                         "extractor run for the paper; default deepseek.")
    ap.add_argument("--llm-sample", type=int, default=150,
                    help="Records per scenario for the LLM study (default 150).")
    ap.add_argument("--llm-model", default=None,
                    help="Model ID (default deepseek-chat).")
    ap.add_argument(
        "--allow-live-llm",
        action="store_true",
        help="Allow DeepSeek API calls on cache misses (disabled by default).",
    )
    args = ap.parse_args()
    seeds = parse_seed_spec(args.seeds)
    base_dir = Path(__file__).parent.parent

    print("=" * 72)
    print("SIGNAL-QUALITY ANALYSIS - extractor phi vs latent yield factor")
    print(f"(noisy note observation; pooled over {len(seeds)} seeds)")
    print("=" * 72)
    print(f"{'Scenario':<28} {'N':>6} {'r(phi,y)':>10} {'r(yhat,y)':>11} {'phi=1.0 frac':>13}")
    print("-" * 72)

    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        res = measure_signal_quality(cfg, key, seeds)
        rphi = f"{res['r_phi']:.3f}" if not np.isnan(res["r_phi"]) else "undefined"
        ryhat = f"{res['r_yhat']:.3f}" if not np.isnan(res["r_yhat"]) else "undefined"
        print(f"{label:<28} {res['n']:>6} {rphi:>10} {ryhat:>11} {res['frac_fallback']:>12.1%}")

    print("-" * 72)
    print("r(phi,y)  = Pearson r between phi and the latent yield factor")
    print("r(yhat,y) = Pearson r between value-weighted component-yield prior and target")
    print("'phi=1.0 frac' = share of assets where the extractor found no signal and")
    print("                 fell back to the uninformative prior (phi=1.0).")

    # ---- Extractor-quality ladder on the synthetic data ----
    # Deterministic references on the same noisy condition-anchored notes:
    #   Keyword       = restricted-vocabulary reproducible reference
    #   Phrase-matcher= complete vocabulary, discrete phi per condition
    # An LLM (run with --llm) typically exceeds the phrase matcher, so the
    # phrase matcher is a reference point, NOT an absolute ceiling.
    print()
    print("=" * 72)
    print("EXTRACTOR-QUALITY LADDER (synthetic benchmark)")
    print("=" * 72)
    print(f"{'Scenario':<28} {'N':>6} {'Keyword r':>12} {'Phrase-matcher r':>18}")
    print("-" * 72)
    for cfg_path, key, label in SCENARIOS:
        cfg = load_config(base_dir / cfg_path)
        kw = measure_signal_quality(cfg, key, seeds, KeywordExtractor(key))
        st = measure_signal_quality(cfg, key, seeds, StrongExtractor(key))
        kw_r = f"{kw['r_phi']:.3f}" if not np.isnan(kw["r_phi"]) else "undefined"
        st_r = f"{st['r_phi']:.3f}" if not np.isnan(st["r_phi"]) else "undefined"
        print(f"{label:<28} {kw['n']:>6} {kw_r:>12} {st_r:>18}")
    print("-" * 72)
    print("Keyword r        = restricted-vocabulary reproducible reference.")
    print("Phrase-matcher r = complete vocabulary, discrete phi per condition.")
    print("Both readers are deterministic; the note observation is corrupted before extraction.")
    print("Run with --llm to add a (typically higher) LLM column.")

    # ---- Optional cache-backed LLM extractor study ----
    if args.llm:
        model = args.llm_model or _DEFAULT_LLM_MODEL[args.llm_provider]
        print()
        print("=" * 72)
        print(f"LLM EXTRACTOR ({args.llm_provider}: {model}) - bounded subsample, cached")
        print("=" * 72)
        cache_dir = base_dir / "outputs" / "llm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'Scenario':<28} {'N':>6} {'LLM r(phi,y)':>14}")
        print("-" * 72)
        for cfg_path, key, label in SCENARIOS:
            cfg = load_config(base_dir / cfg_path)
            cache_path = cache_dir / f"{key}_{args.llm_provider}_{model}.json"
            try:
                res = measure_llm_signal_quality(
                    cfg, key, seeds, args.llm_sample, cache_path, model,
                    args.llm_provider, args.allow_live_llm,
                )
            except RuntimeError as exc:
                ap.error(str(exc))
            rphi = f"{res['r_phi']:.3f}" if not np.isnan(res["r_phi"]) else "undefined"
            print(f"{label:<28} {res['n']:>6} {rphi:>14}")
        print("-" * 72)
        print(f"LLM r on the synthetic benchmark ({args.llm_sample} records/scenario).")
        print("Responses cached under outputs/llm_cache/ for reproducible re-runs.")


if __name__ == "__main__":
    main()
