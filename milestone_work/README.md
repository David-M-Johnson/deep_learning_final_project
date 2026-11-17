# Milestone workspace

Milestone source, completed experiments and two report-ready figures. The
recommended package is [verified_reproduction/](verified_reproduction/README.md),
containing the completed nine-trial study and matched public-release evaluation.
The earlier `base_model/` copy remains as historical work. Data, checkpoints, large prediction outputs and environments
are kept locally and ignored by Git. The two final milestone images are small
are included with their builder, small summary inputs and provenance.

- [verified_reproduction/README.md](verified_reproduction/README.md): canonical verified pipeline, all nine completed models, copied test evaluation and evidence.
- [milestone_figures/README.md](milestone_figures/README.md): two Computer Modern figures, captions and regeneration.
- [base_model/README.md](base_model/README.md): training, evaluation, copied runs,
  source provenance, protocol and complete baseline file inventory.
- [base_model/results/README.md](base_model/results/README.md): historical baseline snapshots; use verified_reproduction for the completed study.
- [data/README.md](data/README.md): Chinese CSI300/CSI800 data and provenance.
- [base_model/figures/README.md](base_model/figures/README.md): report figures.
- `.gitignore`: excludes local environments, caches, data and generated artifacts.
- [../top k/README.md](<../top k/README.md>): implemented sparse-attention experiments.
- [../ranking loss/README.md](<../ranking loss/README.md>): implemented ranking-loss experiments.

## Setup

From the repository root, using Python 3.11:

```sh
python3.11 -m venv milestone_work/.venv
milestone_work/.venv/bin/python -m pip install -r milestone_work/verified_reproduction/requirements.txt
milestone_work/.venv/bin/python -m pip check
milestone_work/.venv/bin/python milestone_work/verified_reproduction/test_checkpoints.py
```

No environment is copied from the original workspace. To reproduce the exact
verified dependency set, install `verified_reproduction/requirements-lock.txt` instead of
`requirements.txt` on a compatible platform. The lock is an Apple Silicon
Python 3.11 snapshot, not a cross-platform dependency lock.

## Read the completed work

- [Validation study](verified_reproduction/outputs/sweep/COMPARISON.md): three seeds for every width.
- [Matched test results](verified_reproduction/reports/PUBLIC_RELEASE_TEST_RESULTS.md): our width-256 seeds versus the supplied reference.
- [Figures](milestone_figures/README.md): clean illustrations, with validation/test selection explicitly distinguished.
- [Audit](verified_reproduction/reports/AUDIT.md): remaining data/benchmark limitations.

Close agreement on the public release is supported; exact workbook, original
proprietary-data and portfolio reproduction are not claimed.

## New runs

From this directory after setup:

```sh
cd verified_reproduction
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" ../.venv/bin/python train.py \
  --width 128 --seeds 0 1 2 --output runs/new-width128
```

Use a new output directory. Historical completed runs are already copied under
`verified_reproduction/runs/controlled/`; do not overwrite or restart them.
The prior `base_model/` launch commands still exist but are legacy entry points.

## Final figures

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python milestone_figures/make_figures.py
```

All six data files are verified identical to the sandbox copies and stored once
under `data/opensource/`. `data/verified_copy_manifest.json` records their hashes.
No code or data depend on running the original sandbox. The copied reports and
run metadata retain historical source paths for provenance.
