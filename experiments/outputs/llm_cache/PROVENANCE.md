# DeepSeek Cache Provenance

The cache files map exact note text to parsed `[phi, sigma]` values produced by
`deepseek-chat` with temperature 0 and JSON response mode. Exact scenario
guidance and message construction are versioned in
`src/s2s/extractors/deepseek.py`.

## Verifiable Coverage

| Scenario | Generated unique notes, seeds 0--29 | Covered | Cache entries |
|---|---:|---:|---:|
| S1 | 27 | 27 | 27 |
| S2 | 4,840 | 4,840 | 8,477 |
| S3 | 43 | 43 | 43 |

The extra S2 entries came from earlier generated runs. The reviewer audit
validates that every evaluation note has a parseable numeric pair within the
allowed ranges.

## Unavailable Historical Metadata

The original live API request date, provider snapshot/version beyond the model
identifier, token usage, dollar cost, latency, retry count, and parse-failure
count were not recorded. They cannot be reconstructed from the response cache,
so the paper makes no claim about them. Repository file or commit timestamps are
not API-call timestamps and should not be interpreted as such.

Run `python scripts/run_reviewer_experiments.py --seeds 0-29 --include-llm`
to regenerate `llm_cache_audit.csv`.
