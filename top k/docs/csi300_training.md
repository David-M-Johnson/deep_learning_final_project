Run the commands below from the `top k/` directory. The shared environment and datasets remain in the repository root.

# CSI300 top-10 training protocol

## Verified input files

User supplied `/Users/brianyin/Downloads/opensource.zip`. Its CSI300 files were
extracted into ignored `data/opensource/`. CSI800 remains in the archive.
The raw sampler index includes context dates before the intended validation/test
start. Do not score those context dates as part of the later split.

| Split | Raw sampler range | Selected prediction range | Dates |
| --- | --- | --- | ---: |
| Train | 2008-01-02--2020-03-31 | 2008-01-02--2020-03-24 | 2974 |
| Validation | 2020-03-20--2020-06-30 | 2020-04-01--2020-06-19 | 54 |
| Test | 2020-06-17--2022-12-30 | 2020-07-01--2022-12-30 | audited at final evaluation |

The last five dates of training and validation are purged to avoid forward labels
crossing splits. This is an explicit deviation from using every released sample.
Keep it identical in dense and sparse comparisons.

Each sampler has 8 time positions, 222 columns, and finite current-day stock
features in its underlying array. Market vectors are checked to be shared across
stocks for each accessed prediction date. Full-window finiteness is checked on
access. Feature column ordering follows the published sampler contract; the pickle
does not retain a feature-name table. This audit does not independently regenerate
features/labels from raw bars or repair the published validation-processing issue.
The samplers report `ffill+bfill`; window filling uses the supplied loader and is
not independently refitted. Labels in open-source files are raw returns. Training
removes nonfinite labels, drops floor(2.5%*N) from each tail and applies a daily
sample-standard-deviation zscore. Validation/test keep all supplied stock inputs
and mask invalid labels for metrics only. No feature scaler is refitted locally.

## Full run

```bash
caffeinate -i ../.venv/bin/python -u -m src.train --output runs/csi300_top10_seed0 --mode topk --k 10 --width 256 --epochs 40 --device cpu --evaluate-test
```

The command must use a new/empty output directory. On non-Mac systems omit
`caffeinate`. CUDA/MPS device selection is explicit and fails if unavailable.
Width 256, temporal heads 4, stock heads 2, dropout .5, beta 5, Adam 1e-5,
float32, value gradient clipping 3, seed 0, shuffled daily batches, CPU threads 4.
All eligible stocks remain together before documented training-label filtering.

Stop at the first epoch whose mean daily training loss is <= .95, otherwise after
40 epochs. Save `last.pt` each epoch and `selected.pt` at completion. Validation
IC/RankIC is diagnostic only and does not select a checkpoint, because the authors
report a validation-processing issue. This retains the reference threshold idea;
the deterministic daily-shuffle sequence and boundary purging are project choices.

Test is evaluated once after training with the selected threshold/last weights.
It must not be used to retune k or stopping after inspection. If later experiments
are informed by its results, disclose exploratory reuse or reserve another holdout.
No inference-time returns are model inputs. Daily IC/RankIC are averaged over finite
(nonconstant) dates; their mean/population-std ratios are unannualized. Undefined
day counts are explicitly recorded. No portfolio metrics are computed.

## Outputs and verification

Ignored run directory contains configuration/data manifest with SHA256 values,
`history.json`, `last.pt`, `selected.pt`, `results.json`, optional
`test_predictions.csv` and `daily_metrics.csv`. Checkpoints contain weights,
optimizer, epoch, CPU Torch RNG and configuration. A general resume CLI is not yet
implemented. The NumPy date permutation is deterministic from seed+epoch.
The first full run also has copied model/runner/dependency source in `source/`.
Training source, metrics and predictions are not automatically uploaded to GitHub.

Twelve correctness tests passed, and a five-date training/validation smoke run
saved checkpoints. That smoke result is not financial evidence. A full top-10 run
was launched on November 15, 2026. Epoch 1 completed in 163.12 seconds, with
training loss 1.04084, diagnostic validation IC 0.08351 and RankIC 0.09048.
These are single-epoch validation results, not test results or an improvement claim.
Consult its `results.json` to determine completion.
A single-seed top-10 result alone does not demonstrate improvement: compare a dense
baseline under this exact protocol and then random-neighbor controls/multiple seeds.

## Sequential top-50 run

`src/queue_run.py` can wait for the active top-10 PID to exit successfully after
writing final test results, then invoke top-50 under the same width, seed, training
budget, loss threshold, data and boundary rules. It writes an ignored queue status
JSON and log in `runs/`. A failed/incomplete top-10 run blocks the queue. This is
a local process that must remain alive, not an external scheduler. Keep the laptop
on with the lid open. Run it under `caffeinate -i` to prevent idle system sleep.
Sequential execution avoids concurrent resource contention; top-50 is predetermined,
not chosen after inspecting top-10 test results.

## Completed runs

Both seed-0 runs completed. Top-10 stopped at epoch 32 (loss 0.948924);
top-50 at epoch 26 (loss 0.948610). Both evaluated 611 test dates.
See `results.csv` and the ignored `runs/` folders. Historical manifests, commands
and source snapshots inside run artifacts are preserved as executed and may refer
to their pre-reorganization paths. They have not been rewritten. These results
do not establish improvement over a matched dense baseline.
