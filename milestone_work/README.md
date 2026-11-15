# Milestone workspace

Clean copy of the dense MASTER experiments, separate from the originals in
`david/reproduction/`. Source code and three named model configurations are
prepared for a later commit. Data, checkpoints, figure outputs and environments
are kept locally and ignored by Git. No commit or push is performed here.

- [base_model/README.md](base_model/README.md): training, evaluation, copied runs,
  source provenance, protocol and complete baseline file inventory.
- [data/README.md](data/README.md): Chinese CSI300/CSI800 data and provenance.
- [base_model/figures/README.md](base_model/figures/README.md): report figures.
- `.gitignore`: excludes local environments, caches, data and generated artifacts.
- `top_k/`: existing empty placeholder; no sparse attention is implemented here.

## Setup

From the repository root, using Python 3.11:

```sh
python3.11 -m venv milestone_work/.venv
milestone_work/.venv/bin/python -m pip install -r milestone_work/base_model/requirements.txt
milestone_work/.venv/bin/python -m pip check
milestone_work/.venv/bin/python milestone_work/base_model/smoke_test.py
```

No environment is copied from the original workspace. To reproduce the exact
verified dependency set, install `base_model/requirements-lock.txt` instead of
`requirements.txt` on a compatible platform. The lock is an Apple Silicon
Python 3.11 snapshot, not a cross-platform dependency lock.

## New runs

```sh
cd milestone_work/base_model
../.venv/bin/python run_model.py --width 64
../.venv/bin/python run_model.py --width 128
../.venv/bin/python run_model.py --width 256
```

Launch each in a separate terminal if running concurrently; they share CPU/RAM.
These are fresh training runs under `base_model/runs/`, not continuation of copied
checkpoints. Add `--dry-run` to inspect a command without starting it.
Copied historical artifacts live separately under `base_model/saved_runs/`.
