# SSADS Paper

`main_7pg.tex` is the canonical camera-ready source and `main_7pg.pdf` is its
compiled seven-page A4 IEEE conference manuscript.

The manuscript is not covered by the repository's MIT License. See
`COPYRIGHT.md` before redistributing or replacing any manuscript file.

Compile with Tectonic:

```bash
tectonic -X compile main_7pg.tex
```

The manuscript numbers come from:

```bash
cd ../experiments
pip install -r requirements-repro.txt
python scripts/run_summary.py --seeds 0-29
python scripts/run_summary.py --seeds 0-29 --extractor strong
python scripts/run_summary.py --seeds 0-29 --extractor llm
python scripts/run_calibration.py --seeds 0-29 --llm
python scripts/run_ablation.py --seeds 0-29
python scripts/run_diagnostics.py --seeds 0-29
python scripts/run_reviewer_experiments.py --seeds 0-29 --include-llm
```

The reproducibility audit covers noisy-full, oracle, and combined comparators;
exact allocation; matched-cost inspection; feature-signal quality; selective
risk; note diversity; parameter, threshold, noise, and cache checks. Its
note-noise variants hold latent assets and underlying noise draws fixed.

The paper cites the public repository root as the reproducibility artifact. The
committed DeepSeek score caches cover the stated 30-seed evaluation; the commands
above run cache-only unless live access is explicitly enabled.
