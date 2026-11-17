# Top-k inter-stock attention experiments

Candidate modification of MASTER: preserve only the k highest-scoring source stocks
per target, head and time position. Dense mode is retained in the same implementation.
Self is eligible but not forced; k includes self if selected. Dense score computation
is still performed, so no computational efficiency gain is claimed.

## Completed results

All rows below are **single-seed (seed 0), width-256 CSI300 open-source results**.
The top-k model keeps all eligible stocks in each date batch and masks source
connections; self is eligible but not forced. Test predictions cover **611 trading
dates, July 1, 2020 through December 30, 2022**.

| Model | Epochs | Final training MSE | Test IC ↑ | Test RankIC ↑ | Test ICIR ↑ | Test RankICIR ↑ | Train + validation (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense MASTER (MSE control) | 26 | 0.9490 | 0.05433 | 0.06513 | 0.32735 | 0.39045 | 60.3 |
| Top-10 | 32 | 0.9489 | 0.04958 | 0.05737 | 0.33977 | 0.39397 | 81.4 |
| Top-50 | 26 | 0.9486 | 0.05571 | 0.06533 | 0.36204 | 0.42705 | 66.1 |

The dense row is the separately trained **MSE-only control** from the ranking-loss
experiment, reusing the same dense MASTER, dataset processing, evaluation and seed.
All three runs use eight-day inputs, 158 stock and 63 market features, four temporal
heads, two stock heads, dropout 0.5, beta 5 and Adam with learning rate 1e-5.
The first epoch reaching mean training MSE <= 0.95 is selected, with a maximum of
40 epochs; validation is diagnostic, not used to select the checkpoint.
The runs share the same chronological splits and five-date boundary purges.
See [the training protocol](docs/csi300_training.md) for details and the published
validation-processing caveat. The two top-k summaries are also in
[results.csv](results.csv); that file does not include the later dense control.

**Interpretation:** top-10 had lower test IC and RankIC than dense MASTER, but
slightly higher ICIR and RankICIR. Top-50 was numerically higher on all four, but
its RankIC advantage was
only **0.00021** (0.06533 versus 0.06513). These single-seed results do **not**
establish a reliable improvement. Multiple seeds, random-neighbor controls and
uncertainty analysis remain necessary. The test period has already been inspected,
so further development must disclose exploratory reuse rather than treat it as a
new untouched holdout.

The durations above are measured wall times for each complete training/validation
run, not comparable fixed-epoch speed tests: stopping epochs and machine conditions
differ. This implementation computes dense scores before masking, and **no speed
or memory improvement is claimed**. No portfolio backtest has been run.

Checkpoints, dated test predictions, daily metrics, manifests and logs are retained
locally under each experiment's ignored `runs/` directory; they are not uploaded
with this README. The authors' five-seed open-source averages are a separate
reference, not a replacement for the matched local dense comparison above.

## Commands

Fresh setup (Python 3.11, macOS/Linux):

```bash
git clone --recurse-submodules https://github.com/David-M-Johnson/deep_learning_final_project.git
cd deep_learning_final_project
python3.11 -m venv .venv
.venv/bin/python -m pip install -r "top k/requirements.txt"
cd "top k"
../.venv/bin/python -m pytest -q
```

For an existing clone, run `git pull` and `git submodule update --init original_paper/MASTER`
from the repository root before running tests. The reference checkout is pinned
by the parent repository. No training data is needed for the 12 correctness tests;
they use generated inputs plus the authors' supplied checkpoint.

On Windows, create the environment with `py -3.11 -m venv .venv` and use
`.venv\Scripts\python.exe` from the root or `..\.venv\Scripts\python.exe`
from this folder. Tests run on CPU; NVIDIA training requires a compatible CUDA
PyTorch installation. Windows/CUDA execution has not been verified here.

A new top-10 run (choose a new output directory):

```bash
../.venv/bin/python -m src.train --mode topk --k 10 --output runs/new_top10_seed0 --evaluate-test
```

Change `--k 50` for top-50 or `--mode dense` for a matched dense run.
Full training additionally requires the authors' CSI300 open-source pickles:
`csi300_dl_train.pkl`, `csi300_dl_valid.pkl`, `csi300_dl_test.pkl`.
Obtain them from the [upstream dataset links](https://github.com/SJTU-DMTai/MASTER#dataset)
and place them in the shared repository-root `data/opensource/` directory.
Default data paths point there; use `--data-dir` for a different location.
The full checkpoints and predictions from our runs are local, not included in GitHub;
`results.csv` contains their summary. See the protocol for exact split/filtering choices.
Install dependencies with `../.venv/bin/python -m pip install -r requirements-data.txt`.
The original checkout stays at `../original_paper/MASTER/` and was not changed.

## File inventory

| Path | Purpose |
| --- | --- |
| `src/__init__.py` | Python package marker. |
| `src/model.py` | Attributed dense and top-k MASTER implementation. |
| `src/train.py` | Training, diagnostic validation, threshold/last checkpoints and optional test scoring. |
| `src/benchmark.py` | Bounded synthetic or trusted real-training-pickle throughput measurement. |
| `src/queue_run.py` | Local sequential top-50 launcher after successful top-10 completion. |
| `tests/test_attention.py` | Dense/reference equivalence, neighbor counts, gradient/update/reload and checkpoint checks. |
| `tests/test_training.py` | Training label processing and inference-stock preservation checks. |
| `pytest.ini` | Local import path and test discovery configuration. |
| `requirements.txt` | Core dependency ranges. |
| `requirements-data.txt` | Additional Qlib dataset dependency. |
| `environment-macos.txt` | Installed Mac environment snapshot; not a CUDA lock. |
| `THIRD_PARTY_MASTER_LICENSE` | Upstream MIT license, SJTU-DMTai/MASTER revision de8f58557096abde4216a701b35fc4368158d111. |
| `docs/attention_benchmarks.md` | Semantics and measured synthetic/real throughput. |
| `docs/csi300_training.md` | Data audit, training protocol, limitations and completed-run notes. |
| `results.csv` | Small, committed-format summary of the two completed runs. |
| `runs/` (ignored) | Full checkpoints, predictions, logs, manifests and source snapshots. |

Historical manifests and copied run source preserve their original commands/paths.
A directory move does not change the measured runs or their archived implementations.
