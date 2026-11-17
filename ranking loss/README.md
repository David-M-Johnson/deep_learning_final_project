# Ranking-aware loss: dense MASTER experiment

## Hypothesis and method

MASTER is trained to regress standardized returns, but evaluation emphasizes stock
ordering. This candidate adds a pairwise logistic ranking term while preserving
dense attention. It is independent of the top-k experiment.

For the retained stocks on each training date:

```
L = MSE(prediction, normalized_return) + lambda * L_pair
L_pair = mean softplus(-sign(y_i-y_j) * (prediction_i-prediction_j))
```

Use all unordered pairs i<j with unequal labels. Exact ties are excluded; if no
usable pairs exist the ranking term is differentiable zero. All input losses must
be finite. MSE maintains score calibration; the ranking term penalizes misordering.
The scale of pairwise differences is the raw model-score scale (temperature 1).
At equal scores, L_pair=log(2). This is a project implementation of a pairwise
logistic ranking objective, not an exact reproduction of a published ranking method.
Learning-to-rank precedent: Feng et al., *Temporal Relational Ranking for Stock
Prediction*, https://arxiv.org/abs/1809.09441.

Initial lambda=0.1 is fixed before these runs; it is not chosen from test results.
The matched control uses lambda=0 (exact MSE, identical gradients). Pairwise loss
uses all eligible training pairs with no subsampling/RNG draws. No top-k masking.
The pairwise computation is quadratic in the daily stock count; at CSI300 it is
roughly 40,000 pairs after training-label filtering, not a new temporal model.

## Shared implementation and provenance

`ranking_experiment/shared.py` imports dense MASTER and the audited data processing
and evaluator from `../top k/src/`. The folder name does not mean sparse attention
is active: the runner explicitly selects `attention_mode="dense"`. Keep both folders
when cloning. No baseline files or completed top-k artifacts are modified.
Underlying MASTER: Li et al., AAAI 2024; SJTU-DMTai/MASTER revision
`de8f58557096abde4216a701b35fc4368158d111`, MIT license preserved in
`../top k/THIRD_PARTY_MASTER_LICENSE`.

## Completed results

CSI300 open-source, width 256, seed 0; both models stopped at epoch 26 when
training MSE reached 0.95. Test covers 611 dates, July 2020--December 2022.
The control in this table is this experiment's matched MSE-only run, not the
separate milestone reproduction pipeline.

| Model | Test IC | Test RankIC | Test ICIR | Test RankICIR | Train + validation minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense MSE control | 0.05433 | 0.06513 | 0.32735 | 0.39045 | 60.3 |
| Dense + ranking loss | 0.05427 | 0.06529 | 0.32704 | 0.39128 | 64.1 |

Ranking loss increased RankIC by only 0.00016, while IC decreased slightly.
**No meaningful improvement was demonstrated in this single-seed comparison.**
The results do not establish that all ranking objectives or weights are ineffective.
See [results.csv](results.csv) for exact values and [epoch_metrics.csv](epoch_metrics.csv)
for the training/diagnostic-validation history. Additional seeds and uncertainty
analysis remain necessary. No portfolio evaluation was performed.

## Trained checkpoints on GitHub

Download the [CSI300 seed-0 checkpoint release](https://github.com/David-M-Johnson/deep_learning_final_project/releases/tag/csi300-seed0-checkpoints-v1). It includes the original
full selected checkpoint for top-10, top-50, top-100, ranking-loss dense MASTER
and its dense MSE control. Each `*-training.pt` file is about 9 MB and contains
model weights, optimizer state, epoch, configuration, RNG state and data manifest.
`training-manifest.json` records file sizes and SHA256 checksums. No datasets are
included, and release files are not downloaded automatically by `git pull`.

Keep the project code and use the matching feature ordering/preprocessing. These
trusted project checkpoints include historical Python metadata, so load them using
`torch.load(path, map_location="cpu", weights_only=False)`. The model weights are
under `model_state_dict`, and training settings are under `config`. Load those
weights into the corresponding MASTER architecture and call `model.eval()` for
prediction. Historical local paths in metadata do not need to match your machine.
The runner does not yet implement an automatic training-resume command.


## Setup and checks

For a fresh clone on macOS/Linux (Python 3.11):

```bash
git clone --recurse-submodules https://github.com/David-M-Johnson/deep_learning_final_project.git
cd deep_learning_final_project
python3.11 -m venv .venv
.venv/bin/python -m pip install -r "ranking loss/requirements.txt"
```

Then, from the repository root:

```bash
cd "ranking loss"
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m pytest -q
../.venv/bin/python -m ranking_experiment.suite --smoke --output runs/new_smoke
```

Twelve objective/integration checks cover ordering, tied labels, equal-score loss,
shift/label-scale invariance, weight-zero loss/gradient equivalence, direction of
ranking gradients, invalid inputs/weights, dense-model updates and checkpoint reload,
and matched experimental configurations. Real-data five-date training/validation
smoke runs completed for ranking and its MSE control. No smoke-test forecast claim.

## Frozen initial full comparison

```bash
caffeinate -i ../.venv/bin/python -u -m ranking_experiment.suite --output runs/paired_seed0 --evaluate-test
```

On systems other than macOS, omit `caffeinate`. The suite runs ranking loss first,
then dense MSE sequentially; it fails rather than running the next stage after a
training failure. It saves a status file, per-run logs and an eventual comparison CSV.
Use `--dry-run` to inspect the exact commands; output directories must be new.
`configs/*.json` are read by the suite; use the train CLI for individual runs.

Both use open-source CSI300 data in repository-root `data/opensource/`, width 256,
8-day history, 158 stock/63 market features, 4 temporal/2 stock heads, beta 5,
dropout .5, Adam 1e-5, value gradient clipping 3, float32, CPU/4 threads, seed 0.
Chronological prediction intervals: 2008-01-02--2020-03-24 training,
2020-04-01--2020-06-19 diagnostic validation, 2020-07-01--2022-12-30 test.
Boundary dates are purged as in the top-k project protocol. Original sampler
feature processing is reused, with missing labels/tails filtered and labels
standardized only for training. Evaluation retains all provided input stocks.

Both stop at mean daily **MSE component <=0.95**, or 40 epochs. The combined loss
has a different scale and must not be compared against the original MSE threshold.
This is a matched threshold-based protocol, not fixed identical epoch counts;
training time and stopping epoch will be reported. Validation is diagnostic only
due to the documented upstream processing caveat. Save last/selected checkpoint,
configuration, source/data hashes, history, prediction CSV and metrics.
The suite saves both ranking and reused shared source before training.

The primary metric is held-out RankIC; IC/ICIR/RankICIR and runtime are secondary.
A single seed is preliminary. No portfolio performance is inferred from these metrics.
The same historical test period was previously inspected for top-k. This new comparison
is exploratory; no lambda sweep or checkpoint selection should use its test results.
Additional seeds and a separate holdout/walk-forward study are needed for strong claims.

## Current status and file inventory

Both full runs completed successfully. Small result summaries and histories are
committed; per-stock predictions, logs, source snapshots, and intermediate checkpoints
remain local in ignored `runs/paired_seed0/`. They are not part of a fresh clone; selected checkpoints are downloadable from the linked release.
For full training, download the authors' open-source CSI300 sampler files and put
`csi300_dl_{train,valid,test}.pkl` in repository-root `data/opensource/`.
Use `--data-dir` with the individual training CLI for another location.
No dataset is needed to run the unit tests (but the sibling `top k/` code is required).
Windows users should replace `../.venv/bin/python` with
`..\.venv\Scripts\python.exe`; Windows/CUDA execution has not been verified here.

| Path | Purpose |
| --- | --- |
| `ranking_experiment/__init__.py` | Distinct Python package; avoids collisions with other experiment suites. |
| `ranking_experiment/shared.py` | Imports shared dense MASTER, dataset loader and metrics. |
| `ranking_experiment/loss.py` | Combined MSE and all-pair logistic ranking objective. |
| `ranking_experiment/train.py` | Configurable ranking-weight runner, component logs and MSE-based stopping. |
| `ranking_experiment/suite.py` | Sequential matched ranking/MSE runs, source snapshots, status and comparison table. |
| `configs/dense_ranking.json` | Prespecified width-256 seed-0 lambda=.1 settings. |
| `configs/dense_mse.json` | Identical settings with lambda=0 for the dense control. |
| `tests/test_loss.py` | Objective and integration correctness checks. |
| `pytest.ini` | Local import path/test discovery. |
| `requirements.txt` | Python dependencies, including Qlib for released sampler pickles. |
| `results.csv` | Exact matched seed-0 validation/test metrics and run durations. |
| `epoch_metrics.csv` | Both runs' per-epoch MSE, ranking term, diagnostic validation metrics and times. |
| `runs/` (ignored) | Smoke/full logs, manifests, histories, checkpoints and generated results. |
