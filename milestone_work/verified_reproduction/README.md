# Verified dense MASTER reproduction

Recommended milestone implementation. This is a self-contained copy of the
completed sandbox verification and three-seed study, sharing the single dataset
directory `../data/opensource/`. The earlier `../base_model/` workspace is kept
as historical work; no prior files or runs were deleted. The sandbox originals
are unchanged. This package does not import or load runtime data from sandbox/.

## What is established

- Widths 64, 128 and 256 each trained with seeds 0, 1 and 2: nine completed trials.
- Reference architecture and tested update mechanics agree exactly when weights,
  batches, threading and dropout RNG are matched. Whole-run RNG and data/split
  differences are documented; this is not a claim of identical proprietary-data training.
- Latest, best-validation-IC and threshold-selected checkpoints are retained
  separately. Optimization and the original loss-threshold stopping rule are unchanged.
- A fixed, matched public-release test evaluation used the authors' supplied
  width-256 checkpoint and our three threshold-selected width-256 checkpoints.
  All models used 611 identical dates, 300 inference stocks per date and the
  authors' unchanged evaluation code. Missing labels affect scoring only.
- Our mean test IC is 0.061261 +/- 0.005297; RankIC 0.068101 +/- 0.003481.
  The supplied reference scores IC 0.065416 and RankIC 0.069103.
- The supplied checkpoint's workbook values are close but outside the declared
  exact-match tolerances. The remaining difference is unresolved. Proprietary
  original-paper data and portfolio backtest results are not reproduced here.

See [study overview](reports/RESULTS_OVERVIEW.md),
[validation comparison](outputs/sweep/COMPARISON.md), and
[matched test results](reports/PUBLIC_RELEASE_TEST_RESULTS.md).

## Setup and commands

From the repository root, if the milestone environment is not already installed:

```sh
python3.11 -m venv milestone_work/.venv
milestone_work/.venv/bin/python -m pip install -r milestone_work/verified_reproduction/requirements.txt
milestone_work/.venv/bin/python -m pip check
```

The full lock is the existing verified Apple Silicon Python 3.11 snapshot,
not a cross-platform lock. A virtual environment is not copied into the package.
From `milestone_work/verified_reproduction/`:

```sh
# Read the completed study:
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python status.py
# Verify copied weights/metrics without selecting new checkpoints:
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" ../.venv/bin/python verify_paired_test.py
# Check atomic checkpoint saving and reload behavior:
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" ../.venv/bin/python test_checkpoints.py
# Bounded new training check (choose a fresh output directory):
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" ../.venv/bin/python train.py \
  --width 64 --max-dates 5 --epochs 1 --output runs/new-smoke
# New full training, explicit width/seeds, separate from preserved trials:
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" ../.venv/bin/python train.py \
  --width 128 --seeds 0 1 2 --output runs/new-width128
```

The completed study and test-evaluation output directories are protected against
accidental overwrite. Do not re-run the sweep into its existing trial paths.
The trainer's generic test flag remains disabled; the separate, frozen
`paired_test_eval.py` implements the explicitly approved test protocol. The test
set has now been evaluated and is not unseen for further adaptive tuning.
No additional full training was started during this copy.

## Model and protocol

Dense MASTER: 158 Alpha158 stock inputs, 63 market inputs, eight observations,
market gate beta 5, four temporal/two stock heads, dropout 0.5, Adam lr 1e-5,
gradient-value clipping at 3, at most 40 epochs. Widths 64/128/256 contain
66,177 / 212,353 / 775,041 parameters. Stop at training MSE <= 0.95, otherwise
at the epoch limit. Best-validation checkpoint selection uses IC with earliest
tie retention and does not alter the stopping decision. Three-seed means and
population standard deviations are reported, retaining individual seed results.

The target is `close[t+5]/close[t+1]-1`. Features and labels follow the supplied
processed release; training labels are filtered and standardized per date.
Training/validation use chronological paper intervals with explicit context
exclusion and five-date boundary purges (2,974 / 54 retained dates). Test uses
2020-07-01–2022-12-30 without final-date purging. All supplied inference stocks
remain in attention; missing labels are masked only for metrics. IC/RankIC are
daily cross-sectional correlations; stability ratios are mean/population std,
without annualization. No transaction-cost or portfolio profitability claim is made.

Data provenance and missing-validation-stock limitations are in
[the audit](reports/AUDIT.md). Processed samplers discarded feature names/raw
price bars, so exact feature-expression ordering, raw target recomputation and
normalizer fitting cannot be independently reconstructed from these files.
The renewed validation filtering issue is disclosed, not claimed repaired.

## Files and provenance

Source research code: https://github.com/SJTU-DMTai/MASTER at revision
`de8f58557096abde4216a701b35fc4368158d111`; `vendor/LICENSE` retains MIT terms.
Tong Li et al., *MASTER: Market-Guided Stock Transformer for Stock Price
Forecasting*, AAAI 2024, 38(1), 162–170, arXiv:2312.15235.

`COPY_MANIFEST.json` records original source hashes, destination hashes and the
relocated source hashes. Source outputs/checkpoints are byte-preserved; only
runtime filesystem addresses and wording are adapted in the relocated code.
`historical_protocols/` retains exact original frozen configs. Current `configs/`
contains equivalent relative paths plus updated relocation code hashes.
Historical absolute paths and PIDs in copied run records describe completed
sandbox executions; they are not current runtime dependencies or active jobs.

All six Chinese data files already existed in `../data/opensource/`; their
SHA256s were checked against the sandbox copies, so no duplicate dataset is stored.
All nine trials' latest/best/threshold checkpoints (where present), full epoch
histories, validation predictions and summaries are copied under `runs/controlled/`.
The author checkpoint, workbook, PDFs and reference code are copied under
`reference/`. Four test prediction/daily-metric sets and the matched report are
under `outputs/public_release_test/`. Data, weights, predictions, caches and
large reference artifacts stay Git-ignored. Source, small reports/configs and
the two milestone figures can be committed; no commit or push was performed.

## Maintained file inventory

| Path | Actual purpose |
| --- | --- |
| `README.md` | Recommended workflow, results scope, source/data provenance and inventory. |
| `reports/COMMAND_VERIFICATION.md` | Fresh-environment README command checks, fixes and platform limits. |
| `COPY_MANIFEST.json` | Verified copy identities, source/destination checksums and relocation hashes. |
| `train.py` | Shared dense trainer for all widths, retention, daily metrics and resource records. |
| `checkpointing.py` | Atomic latest/best-IC/threshold checkpoint saving and tie handling. |
| `baseline_runner.py` | Unmodified historical project runner used in equivalence checks. |
| `verify_equivalence.py` | Actual reference-loop comparisons on nine real width/date cases. |
| `test_checkpoints.py` | All-width checkpoint selection, atomic-save and exact-reload tests. |
| `audit_data.py` | Dataset coverage/missingness/context checks and cached workbook extraction. |
| `benchmark_cpu.py` | Bounded training-only threading/timing probe. |
| `sweep.py` | Frozen nine-trial driver and separate latest/best validation aggregation. |
| `status.py` | Reads copied trial histories and completion state. |
| `paired_test_eval.py` | Shared original metric/predict loop for fixed-checkpoint public test evaluation. |
| `verify_paired_test.py` | Common-population, metric, hash and sampled-prediction checks. |
| `diagnose_reference_context.py` | Reference-only context-date investigation; primary test metrics unchanged. |
| `requirements.txt` | Pinned core Python 3.11 dependencies. |
| `requirements-lock.txt` | Full previously verified Apple Silicon package snapshot. |
| `vendor/master.py` | Unmodified upstream market-gated dense architecture. |
| `vendor/base_model.py` | Unmodified upstream training wrapper, required by model imports. |
| `vendor/LICENSE` | Upstream MIT license and copyright. |
| `configs/controlled_baseline.json` | Relocated frozen width/seed/settings/data/code identities. |
| `configs/public_release_test.json` | Relocated matched test checkpoint list, input hashes, window and tolerances. |
| `historical_protocols/controlled_baseline.json` | Exact original sandbox training protocol. |
| `historical_protocols/public_release_test.json` | Exact original approved sandbox test protocol. |
| `reports/RESULTS_OVERVIEW.md` | Short guide to the completed validation study and matched test evaluation. |
| `reports/PRESERVED.md` | Earlier experiment snapshot status, separate from controlled trials. |
| `reports/EQUIVALENCE.md`, `reports/equivalence.json` | Written findings and measured preprocessing/forward/loss/gradient/update differences. |
| `reports/AUDIT.md`, `reports/audit_findings.json` | Data/validation findings, reference benchmark and unresolved provenance. |
| `reports/data_audit.json` | Dataset identities, dimensions, missingness, sample counts and split details. |
| `reports/*_daily_counts.csv` | Train/validation/test stock counts per prediction date. |
| `reports/benchmark_workbook_cells.json` | Read-only workbook extraction with exact cell references. |
| `reports/renewed_benchmark.csv` | Recalculated five-seed renewed-data benchmark means/std. |
| `reports/checkpoint_tests.json`, `reports/integration_tests.json` | Retention/reload and unchanged-training integration evidence. |
| `reports/source_integrity.json` | Original sandbox source/copy identity checks (historical provenance). |
| `reports/cpu_timing.json`, `reports/mps_timing.json` | Bounded hardware probes, not full-study speedup claims. |
| `reports/PROTOCOL.md` | Original controlled-study rules and resource policy. |
| `reports/PUBLIC_RELEASE_TEST_PROTOCOL.md` | Fixed evaluation protocol and scope. |
| `reports/PUBLIC_RELEASE_TEST_RESULTS.md` | Full same-data test scores, workbook gap and context diagnostic. |
| `reports/paired_test_verification.json` | Verification of copied test populations, weights, metrics and sample predictions. |
| `runs/` (ignored) | Nine completed trial outputs plus bounded relocation checks; regeneration commands above. |
| `outputs/` (ignored) | Completed validation aggregate, test predictions/daily metrics, reports and diagnostic. |
| `reference/` (ignored) | Copied source code, market CSV, PDFs, workbook and supplied seed-0 weights. |
| `.cache/`, `.tmp/`, `__pycache__/` (ignored) | Local caches and temporary verification files. |

## Relocation verification

All six data identities and copied artifact hashes were checked. The relocated
matched-test verifier passes for all four models (common 611 dates/183,300
inference samples, exact aggregate recomputation, sampled prediction agreement).
The copied checkpoints and atomic-selection tests pass at all widths. A two-date
width-256 training smoke test succeeded using only milestone data/output paths.
The nine completed historical trial outputs were not retrained or modified.
