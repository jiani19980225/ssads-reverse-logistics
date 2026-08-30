# RLDB Assumption Provenance

RLDB is a controlled synthetic benchmark. Its exact prices, yields, condition
mixes, labor capacities, inspection costs, observation noise, and note-corruption
rates are modeling assumptions. They were fixed before the reported reruns and
were not statistically estimated from one company, fitted to target recovery
value, or validated as representative industry averages.

Public sources informed the scale and vocabulary of the scenarios:

- FAA Service Difficulty Reports informed the aviation narrative format and
  condition vocabulary: https://www.faa.gov/data_research/aviation_data_statistics/service_difficulty_reports
- National Retail Federation reporting provided retail-return context:
  https://nrf.com/media-center/press-releases/nrf-and-happy-returns-report-2024-retail-returns-total-890-billion
- Backblaze drive statistics provided general storage-device reliability context:
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
- U.S. Bureau of Labor Statistics occupational data provided general labor-cost
  context: https://www.bls.gov/oes/current/oes492094.htm

These sources do not establish the exact YAML values below. The configuration
files, not this context list, are the authoritative scenario specification for
economic, capacity, condition, inspection, and noise assumptions. Generator
mechanics, note templates, extractor vocabularies, and analysis settings live in
versioned source.

## Exact Configuration Summary

| Setting | S1: IT | S2: Aviation | S3: Consumer |
|---|---:|---:|---:|
| Arrivals per batch | 500 | 500 | 1,000 |
| Component price range | \$22--\$1,900 | \$350--\$16,500 | \$3--\$15 |
| Baseline-yield range | 0.650--0.902 | 0.56 | 0.62 |
| Realized-yield Beta concentration | 20 | 20 | 20 |
| Quick inspection | \$19, 15 min | \$50, 20 min | \$10, 3 min |
| Full inspection | \$75, 60 min | \$100, 35 min | \$20, 5 min |
| Quick/full observation SD | 0.15 / 0.05 | 0.15 / 0.05 | 0.15 / 0.05 |
| Component processing | \$40, 32 min | \$600, 150 min | \$12, 10 min |
| Labor capacity | 600 h/week | 800 h/week | 160 h/day |
| Thresholds (low/high) | 0.25 / 0.50 | 0.25 / 0.50 | 0.45 / 0.50 |

S3 additionally assumes whole-unit values of \$120/\$250/\$180 for
smartphone/laptop/tablet and a \$30 rework cost. Complete asset mixes, bills of
materials, component prices, and action costs are in `configs/*.yaml`.

## Synthetic Text and Noise

The generators first sample latent condition and then generate a separately
corrupted observed condition. The reference corruption probabilities are
15% omission and a 25% one-level severity-perturbation attempt. An outward attempt
at an endpoint is clipped and can leave the observed class unchanged. The reviewer
sensitivity suite tests omission from 0--30% and perturbation attempts from 0--40%.
It uses a separate deterministic note stream and fixed per-asset draw count, holding
the latent population and underlying noise draws constant across variants.

The generators contain 17 S1, 20 S2, and 21 S3 base templates. Numeric age,
station, and hour substitutions produce 27, 4,840, and 43 unique rendered notes,
respectively, across seeds 0--29. These rendered-note counts and
vocabulary/length profiles are written to
`outputs/reviewer_experiments/data_profile.csv`.

## Non-Targeting Commitment

The benchmark does not adjust prices, yields, capacities, inspection noise, or
text noise to force SSADS to win. The no-signal comparator currently has higher
keyword TRV in all three scenarios; that unfavorable result is retained. This
also reveals a construct-validity limit: the synthetic dollar objective does not
price certification, latent safety failures, warranty exposure, or most
downstream rework.
