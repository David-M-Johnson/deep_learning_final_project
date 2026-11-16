# Validation/data audit and benchmark interpretation

## Checked directly

The sandbox uses independent copies of the authors' renewed CSI300 pickles.
`data_audit.json` records SHA256, dtype, shape, sample counts, segment boundaries,
missingness and sampled historical index checks. All sampled current inputs
are finite; no within-date differences occur in the 63 market positions.
All supplied current labels are finite. Training has 2,974 prediction dates,
validation 54, and the held-out test file 611. Auditing test coverage did not
produce predictions, correlations or model-selection feedback.

The copied code/YAML/paper specify Alpha158 (158 stock inputs), 63 market inputs,
8 lookback observations, `close[t+5]/close[t+1]-1`, and training-only robust
normalization/clip/fill conventions. The copied architecture actively maps the
last observation's market block to 158 feature gates (CSI300 beta 5), applies
them before the unchanged dense temporal/stock attention, and projects to a
scalar score. Sampled sampler indices do not exceed the prediction date.
The sampler's `ffill+bfill` operates inside the lookback window, not on labels
from a future prediction date; this preserves as-of availability but imputes
historical gaps and must not be mistaken for original observations.

## Qlib compatibility and validation issue

The pickles retain `start_idx` and `end_idx` from an older sampler layout.
Validation stores March 20 context rows despite an intended April 1 segment
start. Installed Qlib 0.9.7 returns the whole stored `data_index`; its current
constructor normally slices objects earlier. Explicit calendar filtering is
therefore necessary for these old objects. It preserves context for history
but excludes 2,392 March context stock/date rows from validation scoring.

The released validation interval contains 59 prediction dates. Five final
observed dates are conservatively purged, leaving April 1–June 19 (54 dates).
Training similarly excludes its final five dates, ending March 24, 2020.
This purge is our documented protocol deviation, not the paper's unchanged
released rows. It prevents forward-return labels crossing into the next segment.

The authors' June 26, 2025 notice says published validation data used learning
rather than inference processing. Missing-label samples were removed before
inter-stock attention; all eligible inference stocks were intended. Our valid
file indeed has no missing labels and has 298–300 stocks per retained date.
This is consistent with the notice but does not establish exactly which stocks
were omitted or why. Missing eligible stocks cannot be recovered from a label-
filtered processed snapshot. The runner does not repeat label-based inference
filtering, but cannot restore rows already absent. **This issue is not repaired.**
It does not alter threshold-based stopping, but may affect validation-based
checkpoint choice. We therefore label validation-selected weights as selections
on the supplied, imperfect validation snapshot, not exact paper replication.

## What cannot be independently confirmed from these files

Qlib discarded the original DataFrame columns when constructing these samplers.
The pickle contains no ordered feature-name list. We can verify dimensions,
position-based gate behavior, shared market blocks and the documented source
contract, but cannot independently prove every Alpha158 expression's order.
The source market CSV lists intended CSI300/CSI500/CSI800 expressions; it does
not recover missing stock feature metadata or guarantee equal renewed values.

Raw closing-price bars, the original eligible-universe history, fitted
normalizer statistics and an executable versioned feature-building manifest
are not included. Actual target arithmetic/trading-day alignment and the
normalizer fit period cannot be independently recomputed from the processed
arrays. Exact recovery would require those raw/provider artifacts. Neither the
shape checks nor training success proves these properties. The comparison will
hold this one audited processed snapshot fixed and disclose these limitations.

## Renewed-data benchmark

`reference/model/performance.xlsx`, Sheet1, contains two different experiments:

- Rows 1–10: original training data, renewed open-source evaluation data.
- Rows 11–20: renewed open-source training and evaluation; Qlib for AR/IR.

The second block is the relevant released-data benchmark. CSI300 seed rows are
B14:G18; means B19:G19; standard deviations B20:G20. Five-seed cached values:

| Metric | Mean | Population std |
| --- | ---: | ---: |
| IC | 0.059205 | 0.007049 |
| RankIC | 0.066921 | 0.003298 |
| ICIR | 0.389246 | 0.055555 |
| RankICIR | 0.430528 | 0.027893 |
| Annualized excess return | 0.286236 | 0.046322 |
| Portfolio IR | 2.354435 | 0.352843 |

XML cached values were extracted without modifying the workbook. Means and
population standard deviations were independently recalculated from five seeds
and agree with the cached aggregates. `benchmark_workbook_cells.json` preserves
cell addresses, and `renewed_benchmark.csv` includes both CSI universes.

These are released test/backtest benchmarks. They cannot be compared as
like-for-like evidence against our 54-date validation peaks. AR/IR also require
matching portfolio execution and costs; this sandbox does not reproduce them.
The workbook alone does not identify data checksums, every stopping epoch or
all hidden internal experiment settings. Exact numerical reproduction therefore
remains unestablished even at width 256.
