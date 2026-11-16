Run the commands below from the `top k/` directory. The shared environment and datasets remain in the repository root.

# Dense and top-k MASTER: local implementation

## Scope and provenance

`src/model.py` adapts SJTU-DMTai/MASTER revision
`de8f58557096abde4216a701b35fc4368158d111` (Li et al., AAAI 2024).
The MIT license is preserved in `THIRD_PARTY_MASTER_LICENSE`.
The reference checkout under `original_paper/` was not modified.
No data pipeline, full fit/validation loop, random/correlation attention, or forecast improvements are claimed.

## Neighbor semantics

Construct `MASTER(..., attention_mode="topk", k=10)` or `k=50`;
`attention_mode="dense"` preserves reference computation.
Top-k is applied to scaled stock-attention logits separately for every head,
lookback position, and target stock. Keep k source stocks, mask other logits
with negative infinity, then softmax over sources. The selected set can differ
by target/head/position and input. All eligible stocks remain in the batch.
Self is eligible and counts toward k if selected, but is not forced.
k >= the current stock count is exactly dense. k must be a positive integer.
Tied scores can have device-dependent tie ordering.
Dropout is applied after softmax as in the reference; training-time row sums
therefore need not be one. Tests inspect pre-dropout weights.

This version computes a dense score matrix before masking: it tests predictive
sparsity, not asymptotic speed or memory savings. No extra trainable parameters
are introduced, and supplied checkpoint keys are preserved. Checkpoint files do
not encode attention mode/k: record these separately in experiment configuration.

## Setup and checks

Use Python 3.11. On this Mac the shared repository-root `.venv` is installed.

```bash
python3.11 -m venv ../.venv
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m pytest -q
```

On Windows use `.venv\Scripts\python.exe` and install a CUDA-enabled PyTorch
build appropriate for the NVIDIA driver before installing the remaining dependencies.
The Mac environment snapshot is not a Windows CUDA environment specification.

Ten tests passed: full-model exact dense/reference equivalence; exact all-neighbor
equivalence; exact selected counts and normalization for k=1/10/50; finite gradients,
parameter updates and checkpoint reload for k=10/50; invalid k rejection; strict
loading of the supplied CSI300 open-source checkpoint.

## Synthetic throughput (November 15, 2026)

Apple M1 Pro, 16 GB shared memory; Python 3.11; PyTorch 2.15.0; float32;
4 CPU threads; 300 synthetic stocks; 8 observations; 221 input features;
width 256; 4 temporal/2 stock heads; dropout .5; Adam 1e-5; seed 0.
Thirty timed date batches per variant after one warmup; weights/optimizer reset
after warmup. Timings include transfers, forward, loss, backward, clipping, Adam,
and scalar loss retrieval; device synchronization surrounds timing.

| Mode | CPU ms/date | MPS ms/date |
| --- | ---: | ---: |
| Dense | 48.42 | 22.44 |
| Top-10 | 58.19 | 24.18 |
| Top-50 | 61.76 | 23.02 |

Single short measurements, not repeated statistical timing estimates. Synthetic
labels are standardized/trimmed using the open-source training policy. Random
inputs are not evidence about IC/RankIC, convergence, real-data loading time,
full training duration, or peak total memory. Dense masking is not faster here.
MPS is available outside the execution sandbox; sandbox checks reported unavailable.

Reproduce with:

```bash
../.venv/bin/python -m src.benchmark --synthetic --dates 30 --width 256 --device cpu --output runs/synthetic_cpu_width256.json
../.venv/bin/python -m src.benchmark --synthetic --dates 30 --width 256 --device mps --output runs/synthetic_mps_width256.json
```

## Real CSI300 benchmark: original training archive

The user supplied `/Users/brianyin/Downloads/DLdataset.zip` (2,771,043,827 bytes).
It contains only `original/csi300_dl_train.pkl` and `original/csi800_dl_train.pkl`;
no validation/test splits are in this archive. Only CSI300 was extracted locally.
The original CSI300 training sampler has 809,609 stock-date samples on 2,979 dates,
2008-01-02 through 2020-03-31. The actual range matches the reference training split.
Original labels are already processed, so the benchmark uses `--label-policy original`.
The first 100 dates (2008-01-02 through 2008-05-30) cover 3.36% of date batches
and have 251--276 eligible stocks/date. This is an early contiguous slice, not a
representative forecast-quality evaluation or repeated timing study.

Width-256 CPU benchmark, 4 threads:

| Variant | Seconds / 100 dates | Extrapolated training compute / epoch |
| --- | ---: | ---: |
| Dense | 4.39 | 130.82 seconds |
| Top-10 | 4.75 | 141.62 seconds |
| Top-50 | 5.04 | 150.09 seconds |

The MPS run measured 7.31 / 4.00 / 4.12 seconds per 100 dates for dense /
top-10 / top-50 respectively. The first dense run may include shape-specific Metal
startup/compilation costs as daily stock counts vary; these single sequential timings
do not establish a sparse speedup. Repeated, order-balanced device measurements are
needed for efficiency claims. CPU timings are the simpler initial planning estimate.

Loading/selecting/checksumming took 6.59 seconds separately. Extrapolation excludes
per-date dataset construction for the full epoch (sample tensors are prepared before
timing), validation, checkpoint I/O, changing daily stock counts and thermal effects.
No forecasting comparison or reusable financial checkpoint was produced.
Real benchmark JSON includes the extracted training-file SHA256.
Validation/test files are still required for prediction-quality evaluation.

Pickle loading executes serialized Python: only use trusted data.

Install the optional loader, then run:

```bash
../.venv/bin/python -m pip install -r requirements-data.txt
../.venv/bin/python -m src.benchmark --data ../data/opensource/csi300_dl_train.pkl --dates 100 --width 256 --device mps --label-policy opensource --output runs/csi300_mps.json
```

For renewed open-source training data, `opensource` removes missing labels,
trims 2.5% from each tail, and standardizes the remaining labels each date.
For original training data that already received those transformations, use
`--label-policy original` to avoid double processing.
The code rejects non-finite processed features rather than silently replacing them.
Input must be chronological indexed sampler data yielding [stocks,8,222].
All stocks for selected dates stay together before documented training-label filtering.
Only bounded training updates are performed; no validation/test scoring or
financial checkpoint is produced. Use the training split, not the held-out test.

The JSON records source checksum, package/device information, dates, universe counts,
configuration, losses, timing, and CUDA peak allocation when applicable. Estimated
training epoch time extrapolates the sampled training dates and excludes full-dataset
loader overhead, validation, checkpoint I/O and differences in stock counts.
Report hardware-specific runtime separately from financial predictive quality.
