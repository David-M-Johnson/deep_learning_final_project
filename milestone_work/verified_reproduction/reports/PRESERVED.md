# Existing experiment preservation

Original files are not moved, edited or executed. Fresh sandbox snapshots live
under `sandbox/milestone_work/base_model/saved_runs/`:

| Width | Copied checkpoint epoch | Source completion | Latest validation IC | Latest RankIC | Historical best IC | Best RankIC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 64 | 35 | latest checkpoint at epoch 35; completion unconfirmed | 0.062588 | 0.073650 | 0.107201 | 0.112182 |
| 128 | 40 | completed, epoch limit | 0.043098 | 0.051254 | 0.114848 | 0.117404 |
| 256 | 26 | completed, training threshold | 0.018806 | 0.032929 | see JSON | see JSON |

`milestone_work/base_model/results/preserved_experiments.json` records every
latest metric, historical maximum and its epoch, model SHA256, original config,
source and copy time. `validation_summary.csv` separates checkpoint scores from
best observed validation scores. Earlier best weights do not exist in the
original runs and are not invented here. Width 256's epoch-26 training MSE was
0.949076, crossing the 0.95 threshold; its original elapsed time was about 84 min.

Copies of source data/provenance/environment manifests accompany checkpoints.
Their absolute paths are historical records; sandbox code loads the independent
local data copies. The outside milestone directory is unchanged by this task.
These old single-seed runs are preserved separately from the new controlled
three-seed sweep, whose checkpoints use improved retention.
