# RLDB Experiments

The Reverse Logistics Decision Benchmark (RLDB) is a synthetic testbed for
inspection targeting and recovery allocation in IT decommissioning, aviation
maintenance, and consumer-electronics returns.

## Operational Timeline

1. Generate the asset and latent condition, then generate a separately corrupted
   textual observation anchored to that condition. Pre-draw a hidden component
   outcome for paired method comparisons.
2. Extract condition factor `phi` and signal-quality score `sigma` from text.
3. Select skip, quick, or full inspection. Perform and charge every selected
   inspection before recovery allocation.
4. Update the condition estimate with a noisy quick/full observation.
5. Rank positive expected processing margins per labor minute and allocate the
   capacity left after inspection and default scrap handling.
6. Reveal the pre-drawn stochastic outcome and compute net recovery value.

The implemented SSADS actions are refurbishment, component recovery, and scrap.
The random-routing comparator may also use partial recovery.

## Metrics

- `TRV`: realized gross recovery less processing, inspection, rework, and disposal
- `RPR`: share of arrivals actually processed for recovery
- `ICS`: full-inspection cost for all arrivals minus actual inspection cost

`RPR` is not demand coverage; RLDB has no demand-realization model.
Configured S3 rework is a realized downstream penalty: its cost enters TRV and
its time enters TPR after allocation, but it is not forecast or reserved in the
first-stage capacity constraint.

## Comparators

- `random`: no inspection, random allocation
- `rule_based`: no inspection, first-in/first-out allocation
- `structured`: gradient boosting on age, asset type, and BOM counts, then noisy
  full inspection and the common allocator
- `combined`: gradient boosting on the structured features plus keyword `phi`,
  `sigma`, and fallback status, then adaptive inspection and the common allocator
- `oracle`: exact condition revelation through full inspection
- `no_signal`: no condition signal, no inspection, common allocator
- `semantic_only`: text-guided inspection with fixed threshold routing in place of
  the margin-ranked allocator
- `ours`: text-guided inspection with the common value allocator

The structured model uses 100 estimators, learning rate 0.05, maximum depth 2,
and an independent 4,000-asset training population generated with seed 99999.

## Reproduce

The committed outputs were regenerated with Python 3.12.13 and the pinned
direct dependencies below. Use `requirements-dev.txt` instead for compatible
development ranges.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-repro.txt

pytest -q

# One command: re-run everything and byte-compare against the committed
# reference (6 console artifacts + 16 audit CSVs). Exits non-zero on any drift.
python scripts/verify_reproduction.py

# Or run the individual scripts:
python scripts/run_summary.py --seeds 0-29
python scripts/run_summary.py --seeds 0-29 --extractor strong
python scripts/run_summary.py --seeds 0-29 --extractor llm
python scripts/run_calibration.py --seeds 0-29 --llm
python scripts/run_ablation.py --seeds 0-29
python scripts/run_diagnostics.py --seeds 0-29
python scripts/run_reviewer_experiments.py --seeds 0-29 --include-llm
```

`run_reviewer_experiments.py` writes:

- inspection counts and percentages for every method
- batch and per-asset value with seed-level variance
- threshold and note-noise sensitivity
- matched-cost inspection targeting
- greedy ranking and exact mixed-integer optimality audits
- selective-risk bins and bad-skip outcome distributions
- held-out vocabulary families, emulating an unwritten condition family
- note length, vocabulary, and condition-class profiles
- complete scenario-parameter and LLM-cache audits
- text-only, structured-only, and combined feature-signal comparisons

## Extractors

The keyword and phrase readers are deterministic and implemented in
`src/s2s/extractors/`. The optional DeepSeek reader uses `deepseek-chat` at
temperature 0 with JSON output. Exact scenario guidance and the full system/user
message construction are in `src/s2s/extractors/deepseek.py`.

The generators contain 17 S1, 20 S2, and 21 S3 base templates. Numeric age,
station, and hour substitutions produce 27, 4,840, and 43 unique rendered notes,
respectively, over seeds 0--29. Committed parsed-score caches cover all of those
rendered notes. The S2 cache contains extra entries from earlier runs.
The original live API date, cost, latency, and parse-failure log were not
recorded; `outputs/reviewer_experiments/llm_cache_audit.csv` reports what can be
verified from the artifact. No API key is needed for the committed evaluation
corpus. Reproduction commands run in cache-only mode and fail on a cache miss.
New live calls require `--allow-live-llm`, `DEEPSEEK_API_KEY`, and
`requirements-llm.txt`.

## Assumptions and Integrity

Core scenario values live in `configs/*.yaml`; generator mechanics, text templates,
and analysis settings live in versioned source. Public reports informed plausible ranges,
but these values are stylized assumptions rather than statistically estimated
facility parameters. The benchmark does not tune prices, yields, capacity, or
noise to force a favorable lift.

Generation, extraction, allocation, inspection, and realized outcomes use named,
isolated RNG streams. Per-asset inspection perturbations and realized component
outcomes are shared across methods within a seed. The textual observation is
corrupted separately but is anchored to the latent condition that drives yield;
`p_mislabel` is a one-class perturbation-attempt probability, and endpoint clipping
can leave a label unchanged. The note-noise sensitivity suite additionally uses a
separate deterministic note stream with a fixed draw count, so every variant shares
the same latent assets and underlying noise draws. The linear mean-value allocator
is analytical; Beta sampling is retained only for realized outcomes.
