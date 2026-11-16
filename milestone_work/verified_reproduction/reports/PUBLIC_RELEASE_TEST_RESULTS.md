# Matched public-release test evaluation

Every model uses the same 611 dates (2020-07-01–2022-12-30), 300 inference stocks per date, 183,300 inference rows and 182,995 labeled rows. Missing labels are masked only by the unchanged upstream correlation code; no inference stocks are removed. Legacy context rows are excluded exactly using the stored sampler segment. No final test dates are purged.

| Checkpoint | Epoch | IC | RankIC | ICIR | RankICIR |
| --- | ---: | ---: | ---: | ---: | ---: |
| authors_opensource_seed0 | not supplied | 0.065416 | 0.069103 | 0.439896 | 0.451023 |
| ours_threshold_seed0 | 26 | 0.060008 | 0.070205 | 0.375272 | 0.431684 |
| ours_threshold_seed1 | 28 | 0.055491 | 0.063195 | 0.402830 | 0.447804 |
| ours_threshold_seed2 | 27 | 0.068284 | 0.070903 | 0.466757 | 0.471164 |

Our three-seed threshold-selected mean ± population std: IC 0.061261 ± 0.005297; RankIC 0.068101 ± 0.003481.

## Supplied checkpoint versus its workbook seed-0 result

| Metric | Measured | Workbook seed 0 | Difference | Within declared tolerance |
| --- | ---: | ---: | ---: | --- |
| IC | 0.06541625 | 0.06533319 | +0.00008305 | False |
| ICIR | 0.43989614 | 0.43773767 | +0.00215846 | False |
| RankIC | 0.06910311 | 0.06890112 | +0.00020200 | False |
| RankICIR | 0.45102298 | 0.44812214 | +0.00290084 | False |

Benchmark values come from renewed train/test block, Sheet1 B14:E14. The means over five authors' seeds are not the target for a single supplied seed-0 checkpoint.

## Scope

This is a one-for-one **evaluation** on the public release. It does not establish identical training or reproduce unavailable original proprietary data. Our three checkpoints were fixed by the pre-existing training-loss threshold, not selected using test performance. The historical training sweep remains unchanged. No portfolio/backtest metrics are claimed.


## Reference-only legacy-context diagnostic

A separate diagnostic evaluated the authors' checkpoint on the eight June context dates and combined those with its primary daily metrics. Including all 619 exposed sampler dates gives IC 0.065092, RankIC 0.068452, ICIR 0.440073 and RankICIR 0.448972. This also fails the predeclared workbook tolerances, so context-row inclusion alone does not explain the mismatch. Those June dates are outside the declared test interval. They were not added to the primary comparison, and our models were not reevaluated or selected under this variant. The remaining small benchmark discrepancy is unresolved; different software/backend arithmetic, an unrecorded export/protocol difference or a checkpoint/workbook mismatch are possibilities, not established explanations.
