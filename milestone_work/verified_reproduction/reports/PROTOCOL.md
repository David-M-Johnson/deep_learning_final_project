# Frozen controlled-baseline protocol

This is a validation-only controlled comparison on the supplied processed CSI300
snapshot, not a claim of exact proprietary-data or portfolio reproduction.
Data/validation limitations in AUDIT.md are explicitly retained and disclosed.
All code, working directories, copied data, models, caches and new outputs stay
under sandbox/. The earlier milestone workspace is not edited.

- Widths: 64, 128, 256; independent seeds 0, 1, 2 for each (nine fresh trials).
- One configuration per width; no validation-guided hyperparameter search.
- Learning rate 1e-5, Adam defaults, dropout 0.5, heads 4/2, market gate beta 5,
  eight observations, all 158 stock + 63 market inputs, dense stock attention.
- Full training split, same 2,974 dates; same 54-date validation snapshot.
- Same daily stock cross-sections, training-only label filtering and zscore,
  five-date forward-label purges, and explicit exclusion of legacy context rows.
- Training stops at first epoch with mean training MSE <= 0.95 or at 40 epochs.
  Validation scores do not alter this stop decision or optimizer trajectory.
- Retain latest.pt semantics as `last.pt`; additionally save `best_validation.pt`
  selected by greatest mean daily validation IC, strict improvement, earliest tie.
  `threshold_selected.pt` is written only if the threshold is reached; otherwise
  the record explicitly says absent. It is not fabricated at the epoch limit.
- Validate loaded best weights after fitting. Report latest and validation-best
  IC/RankIC/ICIR/RankICIR separately; best IC selects one checkpoint for all metrics.
- Aggregate mean and population standard deviation across the same three seeds.
  Per-seed metrics and epochs remain available. No claim of significance or
  multi-seed test performance is made from selected validation scores.
- CPU float32, two intra-op threads per job, at most two concurrent jobs.
  Concurrent wall times have contention; record CPU seconds and peak process RSS
  as well. Bounded standalone timing probes are recorded separately. No GPU costs.
- Record dataset SHA256, exact source code hashes, config, seed, package versions,
  split dates, per-epoch metrics, checkpoint selection and stopping reason.
- The held-out test set is disabled in the sandbox runner. Inspecting its shape
  and coverage during data audit does not select models using its outcomes.

## Gates and timing

Nine reference-batch equivalence cases, all-width checkpoint reload/selection
tests, and a two-epoch real-data baseline-vs-new-runner comparison must pass
before sweep launch. Audit findings must exist and code hashes must match.
The frozen machine-readable file is configs/controlled_baseline.json.

Standalone 20-date CPU probes suggest roughly 72/97/154 seconds per epoch at
widths 64/128/256 with two threads, excluding normal validation overhead.
Nine full trials are therefore a **several-hour** job; concurrent execution and
threshold stops make an exact finish time uncertain. No shorter dataset, smaller
epoch budget, reduced feature set, or reused training checkpoint is substituted.
The driver persists status, logs and partial aggregates automatically, including
failures. It never silently overwrites/resumes an existing experiment directory.

## Interpretation and later test evaluation

Validation-selection addresses checkpoint loss but does not repair upstream
missing-stock validation filtering or guarantee generalization. The goal here
is to compare widths consistently and establish reproducibility, not tune until
validation matches a known test benchmark. The original threshold-selected runs
remain separately preserved. Any future test evaluation should use a frozen
checkpoint-selection protocol and report both reproduction deviations and seeds.
