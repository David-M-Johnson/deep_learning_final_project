# Top-k inter-stock attention experiments

Candidate modification of MASTER: preserve only the k highest-scoring source stocks
per target, head and time position. Dense mode is retained in the same implementation.
Self is eligible but not forced; k includes self if selected. Dense score computation
is still performed, so no computational efficiency gain is claimed.

## Completed results

See [results.csv](results.csv): seed-0 CSI300 open-source runs at width 256.
Top-10: test IC 0.04958, RankIC 0.05737 (32 epochs).
Top-50: test IC 0.05571, RankIC 0.06533 (26 epochs).
These are single-seed results, not evidence of improvement over a matched dense baseline.

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
