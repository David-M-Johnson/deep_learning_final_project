# MASTER: Reproduction, Sparse Attention, and Ranking Loss

We study stock-return forecasting using Li et al.'s
[*MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*](original_paper/original_paper.pdf)
(AAAI 2024). We reproduced a dense baseline on the authors' renewed public
Chinese data and implemented two extensions: **top-k inter-stock attention** and
**a pairwise ranking loss**. These are working prototypes; an improvement over
the dense baseline has not been established.

## Team contributions

| Member | Contribution |
| --- | --- |
| Marvin Widjaja (mw2375) | Data preprocessing, attempted S&P 500 adaptation, and report contributions. |
| David Johnson (dmj68) | Reproduced baseline results and led report writing. |
| Brian Yin (by336) | Implemented top-k attention and the ranking-loss method, and contributed to the report. |

## Dataset and goal

We use CSI300, a Chinese stock-market index, with training data from 2008–2020
and test data from July 2020–December 2022. We reproduce MASTER and investigate
whether top-k inter-stock attention and a ranking loss improve stock-return forecasting.

See [data download instructions](milestone_work/data/README.md).
Dense reproduction loads data from `milestone_work/data/opensource/`; extension
runners default to `data/opensource/` and accept `--data-dir` to select another location.

## Evaluation and preliminary results

Metrics are calculated across stocks each day, then summarized over test dates:

| Metric | Meaning |
| --- | --- |
| IC | Mean daily Pearson correlation between predicted scores and realized returns. |
| RankIC | Mean daily Spearman correlation: agreement between predicted and realized stock rankings. |
| ICIR / RankICIR | Mean daily correlation divided by its population standard deviation; not annualized. |
| MSE | Squared prediction error against standardized returns; lower is better. |

Higher IC and RankIC are better; lower MSE is better.

- **Dense MASTER:** trained widths 64, 128 and 256 with three seeds each.
  Our width-256 mean test IC / RankIC were **0.0613 / 0.0681**, close to
  **0.0654 / 0.0691** from the authors' checkpoint under the same evaluation.
- **Top-k attention:** completed runs with k = 10, 50 and 100.
  See [results](<top k/results.csv>).
- **Ranking loss:** tested against an MSE-only control, with similar performance.
  See [results](<ranking loss/README.md#completed-results>).

Extension results are preliminary single-seed experiments. Top-100 was explored
after inspecting earlier test results. Details are in the
[dense study overview](milestone_work/verified_reproduction/reports/RESULTS_OVERVIEW.md)
and [matched test report](milestone_work/verified_reproduction/reports/PUBLIC_RELEASE_TEST_RESULTS.md).

## Code and running it

| Location | Purpose |
| --- | --- |
| [verified_reproduction/](milestone_work/verified_reproduction/README.md) | Canonical dense training, checkpoint handling, evaluation and verification. Architecture: [vendor/master.py](milestone_work/verified_reproduction/vendor/master.py). |
| [top k/](<top k/README.md>) | Dense/top-k architecture, extension training and correctness tests. |
| [ranking loss/](<ranking loss/README.md>) | Pairwise ranking objective and matched MSE-control experiments; reuses the top-k package's dense model. |
| [milestone_figures/](milestone_work/milestone_figures/README.md) | Report figures, captions, builder and plotted-value provenance. |
| [original_paper/](original_paper/) | Read-only paper and upstream reference implementation. |

`milestone_work/base_model/` is an earlier baseline copy. `sandbox/` and
`david/reproduction/` are local development work, not the submission entry points.
Detailed source inventories and experiment protocols live in the linked READMEs.

Commands below use Python 3.11 on macOS/Linux. Start each block from the
repository root. Download and extract the three CSI300 pickle files to
`milestone_work/data/opensource/` before training. Use a new output directory
when repeating a run. Fresh-install and smoke-test checks are documented
[here](milestone_work/verified_reproduction/reports/COMMAND_VERIFICATION.md).

Initialize the pinned reference checkout and install the dense environment:

```bash
git submodule update --init original_paper/MASTER
python3.11 -m venv milestone_work/.venv
milestone_work/.venv/bin/python -m pip install -r milestone_work/verified_reproduction/requirements.txt

cd milestone_work/verified_reproduction
# Check checkpoint saving and reloading; no dataset required.
../.venv/bin/python test_checkpoints.py
# After downloading data: short training check.
../.venv/bin/python train.py --width 64 --max-dates 5 --epochs 1 --output runs/new-smoke
# Full width-256 baseline; choose a new output directory.
../.venv/bin/python train.py --width 256 --seeds 0 1 2 --output runs/new-width256
```

**Top-k and ranking-loss setup.** From the repository root, create a separate
extension environment (Python 3.11). This installs dependencies for both methods:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r "ranking loss/requirements.txt"
```

The commands below reuse the CSI300 files in `milestone_work/data/opensource/`.
Choose new output directories for each run.

**Top-k attention.** From the repository root:

```bash
cd "top k"
../.venv/bin/python -m pytest -q
# Short training check.
../.venv/bin/python -m src.train --mode topk --k 10 \
  --data-dir ../milestone_work/data/opensource \
  --limit-dates 5 --epochs 1 --output runs/new-smoke
# Full training and final test evaluation.
../.venv/bin/python -m src.train --mode topk --k 10 \
  --data-dir ../milestone_work/data/opensource \
  --output runs/new-top10-seed0 --evaluate-test
```

Use `--k 50` or `--k 100` for other neighbor counts, or `--mode dense` for a
matched dense control. Defaults are width 256 and seed 0.

**Ranking loss.** From the repository root:

```bash
cd "ranking loss"
../.venv/bin/python -m pytest -q
# Short training check.
../.venv/bin/python -m ranking_experiment.train --ranking-weight 0.1 \
  --data-dir ../milestone_work/data/opensource \
  --limit-dates 5 --epochs 1 --output runs/new-smoke
# Full ranking-loss model and matched MSE-only control.
../.venv/bin/python -m ranking_experiment.train --ranking-weight 0.1 \
  --data-dir ../milestone_work/data/opensource \
  --output runs/new-ranking-seed0 --evaluate-test
../.venv/bin/python -m ranking_experiment.train --ranking-weight 0 \
  --data-dir ../milestone_work/data/opensource \
  --output runs/new-mse-seed0 --evaluate-test
```

Ranking loss uses dense attention and adds a pairwise ranking term to MSE;
weight `0` gives the MSE-only control. Both stop on the MSE component at `0.95`,
or after 40 epochs. Test evaluation occurs after training.
See the [top-k protocol](<top k/README.md>) and
[ranking-loss protocol](<ranking loss/README.md>) for details.

## Milestone figures

- [Three-seed validation comparison](milestone_work/milestone_figures/01_three_seed_validation.png)
- [Matched width-256 test comparison](milestone_work/milestone_figures/02_matched_test_reproduction.png)

## Download trained models

The original full selected checkpoints for top-10, top-50, top-100, ranking loss
and its dense MSE control are attached to the [CSI300 seed-0 release](https://github.com/David-M-Johnson/deep_learning_final_project/releases/tag/csi300-seed0-checkpoints-v1).
Each `*-training.pt` file is about 9 MB and includes weights, optimizer state and
training metadata. The release includes a checksum manifest and loading instructions.
Datasets remain separate; checkpoints are not downloaded automatically by `git pull`.

## References

- Tong Li et al. *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*. AAAI 2024, 38(1), 162–170. [Paper](original_paper/original_paper.pdf); [arXiv](https://arxiv.org/abs/2312.15235).
- [Official MASTER implementation](https://github.com/SJTU-DMTai/MASTER), revision `de8f58557096abde4216a701b35fc4368158d111`, under the [MIT license](milestone_work/verified_reproduction/vendor/LICENSE).

---

This README was AI-generated based on specific instructions provided by the project team.
