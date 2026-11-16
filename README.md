# MASTER milestone work

This project explores **sparse inter-stock attention and potentially other
modifications** to MASTER for stock-return forecasting. It builds on Li et al.,
[*MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*](original_paper/original_paper.pdf),
AAAI 2024. The milestone includes dense models at widths **64, 128, and 256**,
each trained with three seeds, and a matched test comparison against the
authors' supplied width-256 model.

## Dataset

We use the authors' renewed Chinese **CSI300/CSI800** dataset. The completed
milestone experiments use **CSI300**: training spans 2008–March 2020,
validation April–June 2020, and testing July 2020–December 2022. Boundary
exclusions and preprocessing are documented in the [data audit](milestone_work/verified_reproduction/reports/AUDIT.md).

Local files live in `milestone_work/data/opensource/`, named
`{csi300,csi800}_dl_{train,valid,test}.pkl`. Data and model checkpoints are
Git-ignored and are not included in a fresh clone. See
[download instructions and provenance](milestone_work/data/README.md).

## Project Goal

Establish a reproducible dense MASTER baseline, then investigate whether sparse
attention or other model changes improve forecasting quality or computational
efficiency. Compare methods using consistent data, tuning budgets, seeds and
evaluation rules. The separate [top-k experiments](<top k/README.md>) explore
restricting attention to selected neighbors; correlation-based and random-neighbor
controls remain planned. No sparse-attention improvement over a matched dense
baseline has been established.

## Evaluation Plan

Use chronological train/validation/test splits and report variation across seeds.
Choose checkpoints before test evaluation and distinguish validation-selected
scores from held-out test results.

| Metric | Meaning |
| --- | --- |
| **IC** | Daily Pearson correlation between predicted scores and realized returns across stocks, averaged over dates. Higher positive values are better. |
| **RankIC** | Daily Spearman correlation between predicted and realized stock rankings, averaged over dates. Higher positive values are better. |
| **ICIR / RankICIR** | Mean daily correlation divided by its daily standard deviation; measures signal strength relative to variability. Our implementation does not annualize these ratios. |
| **MSE** | Mean squared error against standardized return targets; the training objective. Lower is better. |
| **Annualized excess return / portfolio IR** | Return above a benchmark and its risk-adjusted consistency. These require a separate backtest with explicit trading costs and portfolio rules and are not implemented in the milestone evaluation. |

Correlation scores are **not accuracy percentages or investment returns**.
The matched public-release evaluation shows close agreement with the supplied
reference model; exact workbook values and original proprietary-data results
have not been fully reproduced. See the [results overview](milestone_work/verified_reproduction/reports/RESULTS_OVERVIEW.md)
and [matched test results](milestone_work/verified_reproduction/reports/PUBLIC_RELEASE_TEST_RESULTS.md).

## Reproduction Code and Training

The training code used for the verified study is
[milestone_work/verified_reproduction/train.py](milestone_work/verified_reproduction/train.py).
The architecture is in [vendor/master.py](milestone_work/verified_reproduction/vendor/master.py).
Use Python 3.11 and follow the [environment instructions](milestone_work/verified_reproduction/README.md#setup-and-commands)
before running these commands from the repository root:

```bash
cd milestone_work/verified_reproduction

# Short workflow check.
../.venv/bin/python train.py --width 64 --max-dates 5 --epochs 1 --output runs/new-smoke

# Full CSI300 training with three seeds.
../.venv/bin/python train.py --width 128 --seeds 0 1 2 --output runs/new-width128
```

Choose width **64**, **128**, or **256** and a new output directory. Defaults are
CPU, learning rate `1e-5`, dropout `0.5`, and at most 40 epochs, stopping earlier
at training MSE `0.95`. Use `--device mps` for a supported Mac GPU or
`--device cuda` for an NVIDIA GPU.

Completed checkpoints and histories are stored locally under
`milestone_work/verified_reproduction/runs/controlled/width{64,128,256}-seed{0,1,2}/`.
The three-width validation comparison uses `best_validation.pt`; the matched
width-256 test comparison uses `threshold_selected.pt`. The shared test-evaluation
code is [paired_test_eval.py](milestone_work/verified_reproduction/paired_test_eval.py).
See the [implementation README](milestone_work/verified_reproduction/README.md)
for protocol details and the maintained source-file inventory.

## Milestone Figures

The two report figures and their builder live in
[milestone_work/milestone_figures/](milestone_work/milestone_figures/README.md):

- [Three-seed validation comparison](milestone_work/milestone_figures/01_three_seed_validation.png)
- [Matched width-256 test comparison](milestone_work/milestone_figures/02_matched_test_reproduction.png)

Regenerate them from the locally copied completed-study results, from the
repository root:

```bash
milestone_work/.venv/bin/python milestone_work/milestone_figures/make_figures.py
```

## References

- Tong Li et al. *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*. AAAI 2024, 38(1), 162–170. [Local paper](original_paper/original_paper.pdf); [arXiv:2312.15235](https://arxiv.org/abs/2312.15235).
- [Official MASTER implementation](https://github.com/SJTU-DMTai/MASTER), revision `de8f58557096abde4216a701b35fc4368158d111`, under the [MIT license](milestone_work/verified_reproduction/vendor/LICENSE). The [local reference checkout](original_paper/MASTER/README.md) is read-only source material.
- [Submitted project proposal](project_proposal.pdf).
