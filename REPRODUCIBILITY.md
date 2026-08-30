# Reproducibility Manifest

This manifest pins what the committed artifacts are, how they were produced, and
how to confirm a rerun matches them. It covers the seven-page manuscript
`paper/main_7pg.pdf` and every table in it.

## Environment

| Item | Value |
|---|---|
| Reference interpreter | Python 3.12.13 |
| Direct dependencies | `experiments/requirements-repro.txt` (exact pins) |
| Hardware requirement | None beyond a CPU; no GPU is used |
| Network requirement | None; LLM rows replay from the committed caches |
| Evaluation seeds | `0-29` (30 paired seeds) |

Transitive dependencies are resolved by pip and are not pinned. The bundle has
also been rerun end to end on Python 3.14.6 with the same pinned direct
dependencies, and every output below was byte-identical.

## One-command check

From `experiments/`, after installing the pinned dependencies:

```bash
python scripts/verify_reproduction.py
```

This re-runs every reported experiment and byte-compares the result against the
committed reference: the six console-output artifacts in `outputs/reference/`
(which carry Table II, Table III, the ablation, and the diagnostics) and the 16
audit CSVs in `outputs/reviewer_experiments/`. It exits 0 only if all 22 match,
so reproduction is a pass/fail check rather than a manual comparison against the
PDF. `--update` regenerates the reference instead of checking it.

## Commands

The individual scripts, run from `experiments/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt

pytest -q                                                    # 81 tests
python scripts/run_summary.py --seeds 0-29                    # Table II
python scripts/run_summary.py --seeds 0-29 --extractor strong # SSADS-Phrase
python scripts/run_summary.py --seeds 0-29 --extractor llm    # SSADS-DeepSeek
python scripts/run_calibration.py --seeds 0-29 --llm          # Table III
python scripts/run_ablation.py --seeds 0-29                   # matched-cost
python scripts/run_diagnostics.py --seeds 0-29                # error/sensitivity
python scripts/run_reviewer_experiments.py --seeds 0-29 --include-llm
```

From the repository root, `./verify_integrity.sh` runs the syntax, Ruff, Mypy,
test, and one-seed end-to-end checks.

## Output digests

`scripts/verify_reproduction.py` checks everything below automatically; the
digests are listed so a reviewer can also spot-check by hand.
`run_reviewer_experiments.py --seeds 0-29 --include-llm` regenerates the files
below. A matching rerun reproduces these SHA-256 digests exactly. Digests are
truncated to 16 hex characters for readability; recompute in full with
`shasum -a 256 experiments/outputs/reviewer_experiments/*.csv`.

| `outputs/reviewer_experiments/` | SHA-256 (truncated) |
|---|---|
| `bad_skip_outcomes.csv` | `e402e607c216ddeb` |
| `condition_mix.csv` | `7f19645f9f011b79` |
| `data_profile.csv` | `e537abec856bd3b1` |
| `exact_optimality_gap.csv` | `a6603b133faece08` |
| `feature_signal_quality.csv` | `71dde7012efdc0a5` |
| `greedy_ranking_ablation.csv` | `c415d06b65e9c828` |
| `held_out_vocabulary.csv` | `9a4fd131a100490e` |
| `inspection_depth_by_method.csv` | `5758bf3a41c6ff7c` |
| `llm_cache_audit.csv` | `7911ae31c4994c55` |
| `matched_cost_inspection.csv` | `a633977139d97154` |
| `note_noise_sensitivity.csv` | `bfe3f9710e5ccbd4` |
| `per_asset_value_by_method.csv` | `cf0d4af2e836cda7` |
| `scenario_parameter_summary.csv` | `796b93e48d13061c` |
| `sigma_selective_risk_bins.csv` | `cca3bad12be9c288` |
| `sigma_selective_risk_summary.csv` | `fe19e25be3afbcde` |
| `threshold_sensitivity.csv` | `9e87d99e05df3315` |

Console-output artifacts, in `outputs/reference/`, carry Table II, Table III,
the ablation, and the diagnostics. They are byte-stable under the pinned
dependencies and are compared by `verify_reproduction.py`:

| `outputs/reference/` | Produced by |
|---|---|
| `summary_keyword.txt` | `run_summary.py --seeds 0-29` (Table II) |
| `summary_phrase.txt` | `run_summary.py --seeds 0-29 --extractor strong` |
| `summary_deepseek.txt` | `run_summary.py --seeds 0-29 --extractor llm` |
| `calibration.txt` | `run_calibration.py --seeds 0-29 --llm` (Table III) |
| `ablation.txt` | `run_ablation.py --seeds 0-29` |
| `diagnostics.txt` | `run_diagnostics.py --seeds 0-29` |

The DeepSeek parsed-score caches are inputs, not regenerated outputs:

| `outputs/llm_cache/` | SHA-256 (truncated) |
|---|---|
| `s1_deepseek_deepseek-chat.json` | `fb888ca3fe521b32` |
| `s2_deepseek_deepseek-chat.json` | `f562f6122e2601f5` |
| `s3_deepseek_deepseek-chat.json` | `e0f28a6b6134293d` |

## What is and is not reproducible

Reproducible from this repository with no network access:

- Every number in the manuscript's tables, text, abstract, and conclusion. All 115 were re-derived from a pristine export of this tree and matched.
- The 30-seed asset populations, note text, inspection decisions, and realized
  outcomes, via named and isolated RNG streams.
- The cached DeepSeek rows, which replay parsed `[phi, sigma]` pairs by exact
  note text.

Not recoverable:

- The original DeepSeek API call date, provider backend snapshot, token usage,
  dollar cost, latency, retry count, and live parse-failure count. These were
  not recorded and cannot be reconstructed from the response cache. Repository
  file and commit timestamps are not API-call timestamps. The cache therefore
  replays parsed scores only, not those deployment attributes.
- Bit-identical LLM output for any note absent from the caches, because that
  requires a live call against a model version that is no longer pinned.

## Scope of the evidence

RLDB is a fully synthetic benchmark. Its prices, yields, capacities, inspection
costs, observation noise, and note-corruption rates are disclosed modeling
assumptions, not facility estimates, and were fixed before the reported reruns.
Every reader is developed and evaluated inside the same generator and template
families, so these artifacts do not establish out-of-family linguistic
generalization or field recovery gains. See
`experiments/calibration/ASSUMPTION_PROVENANCE.md` for assumption provenance and
the manuscript's Limitations subsection for the full statement.
