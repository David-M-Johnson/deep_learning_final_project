# Collected CSI300 dense MASTER results

Refreshed November 15, 2026. These are seed-0 **validation results**, using
54 dates in Q2 2020 after the documented boundary exclusions. No held-out
test predictions or portfolio backtests were computed during this collection.
Original experiment directories under `david/reproduction/runs/` are preserved.

## What was present and what changed

Before this refresh, milestone checkpoint snapshots were at epochs 35, 25 and 5
for widths 64, 128 and 256. Width 64 is still at epoch 35 in the source;
width 128 has now completed 40 epochs. Their checkpoint/history snapshots are
refreshed under `../saved_runs/csi300-width64/` and `../saved_runs/csi300-width128/`.
Earlier versions were retained under `../saved_runs/archive/`.

Each refreshed snapshot contains:

- `seed-0/last.pt`: actual latest model weights, optimizer, epoch and RNG metadata.
- `seed-0/history.csv`: all available epoch-by-epoch training and validation metrics.
- `seed-0/valid_predictions.csv`: full validation stock scores and realized labels.
- `seed-0/valid_daily_metrics.csv`: daily IC, RankIC and number of scored stocks.
- `seed-0/valid_metrics.json`: final aggregate validation scores.
- `seed-0/evaluation_manifest.json`: checkpoint/data paths, checksums and split dates.
- `result_summary.json`, `snapshot.json`, configuration and historical provenance.
- Source completion summaries for width 128, whose epoch agrees with its checkpoint.

Copied checkpoints were strictly loaded into the matching models. Validation
was recomputed across all 54 dates using each copied checkpoint; all four
aggregate metrics match the final logged values within 1e-6.

## Latest available checkpoints

| Width | Checkpoint epoch | Status | Training MSE | Validation IC | RankIC | ICIR | RankICIR |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 35 | Completion unconfirmed; no source completion summary | 0.974710 | 0.062588 | 0.073650 | 0.492436 | 0.605515 |
| 128 | 40 | Finished at epoch limit | 0.956296 | 0.043098 | 0.051254 | 0.424858 | 0.527714 |

These weights are the latest checkpoints, not the best validation checkpoints.
The best observed IC/RankIC for both widths occurred at epoch 6:

| Width | Best observed validation IC | Best observed RankIC |
| --- | ---: | ---: |
| 64 | 0.107201 | 0.112182 |
| 128 | 0.114848 | 0.117404 |

Those earlier weights were overwritten by the original training loop. Best
epoch metrics remain in the histories but do not imply an available model.
Falling training error alongside declining validation correlation is consistent
with overfitting; these curves alone do not establish its cause.

## Width 256 status

The **source width-256 run finished at epoch 26**, stopping because training MSE
reached 0.949076, below the reference threshold of 0.95. Its run summary reports
5,033.98 seconds (about 83.9 minutes). Final validation metrics are:

- IC: 0.018806; RankIC: 0.032929.
- ICIR: 0.176626; RankICIR: 0.282525.

`width256_status.json` records these source-run results. The existing milestone
width-256 checkpoint is **still the earlier epoch-5 snapshot**; it was not
replaced as part of the requested 64/128 refresh. Do not assign epoch-26 scores
to the copied epoch-5 model or compare its older figure as a final-run result.
The source checkpoint remains at `david/reproduction/runs/csi300-width256/seed-0/last.pt`.

## Figures

`../figures/width64/` and `../figures/width128/` have refreshed PDF/PNG learning
curves, captions, and exact history snapshots. `../figures/comparison/` now
compares only the refreshed 64/128 histories, avoiding the stale width-256 copy.
To recreate these plots from `milestone_work/base_model/` after environment setup:

```sh
../.venv/bin/python plot_publication.py --width 64
../.venv/bin/python plot_publication.py --width 128
../.venv/bin/python plot_epochs.py saved_runs/csi300-width64 saved_runs/csi300-width128
```

IC is daily cross-stock Pearson correlation; RankIC is Spearman correlation.
Stability ratios divide their mean by population standard deviation without
annualization. MSE is training squared error against standardized labels.
Higher positive validation correlations are better; none is an accuracy
percentage, portfolio return, or an independent multi-seed estimate.

## File inventory and Git

| File | Purpose |
| --- | --- |
| `README.md` | Snapshot status, metric tables, interpretation and figure commands. |
| `validation_summary.csv` | Latest 64/128 metrics plus best observed IC/RankIC and their epochs. |
| `epoch_metrics.csv` | All 35 width-64 and 40 width-128 logged epochs, including elapsed seconds. |
| `validation_results.json` | Detailed result metadata, best metric epochs and model SHA256 checksums. |
| `width256_status.json` | Current source 256-run completion and metrics, with stale milestone-copy status explicit. |

These small result exports can be committed with the code. Checkpoints,
predictions, copied datasets, archived snapshots, environments and generated
figures remain Git-ignored. No commit or push was performed during collection.
