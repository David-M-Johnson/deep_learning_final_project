# README command verification

Verified on macOS 26.6.2, Apple Silicon, Python 3.11.17, CPU.
Two new virtual environments were installed from the exact requirement files
named in the root README; the existing sandbox environment was not used.

| Check | Result |
| --- | --- |
| Dense dependency installation | Passed; `pip check` reports no broken requirements. |
| Extension dependency installation | Passed; `pip check` reports no broken requirements. |
| Dense `test_checkpoints.py` | Passed at widths 64, 128 and 256. |
| Top-k `python -m pytest -q` | 13 passed. |
| Ranking-loss `python -m pytest -q` | 12 passed. |
| Dense README five-date, one-epoch smoke command | Completed and saved checkpoints. |
| Top-k README five-date, one-epoch smoke command | Completed and saved selected/last checkpoints. |
| Ranking README five-date, one-epoch smoke command | Completed and saved selected/last checkpoints. |
| Ranking MSE-only control, five dates / one epoch | Completed with ranking weight zero. |
| Extension test-data loading, checkpoint reload and evaluation | Passed on two real test dates for both smoke models. Diagnostic execution check only. |
| README figure command | Passed in the fresh dense environment. |
| Isolated figure generation | Passed with only builder + included summary JSONs; no datasets, checkpoints or old outputs. |
| README local link targets and code fences | Checked. |

Dense core versions: NumPy 1.26.4, pandas 1.5.3, SciPy 1.14.1,
PyTorch 2.5.1, Matplotlib 3.9.4, Qlib 0.9.7.
Extension core versions: NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1,
PyTorch 2.15.0, pytest 9.2.0, Qlib 0.9.7.

## Fixes made during verification

The figure builder depended on ignored `verified_reproduction/outputs/` files.
Small, frozen summaries are now included under `milestone_figures/inputs/`,
and the builder reads them directly. The README states the working-directory,
Python/platform, data-download and fresh-output-directory prerequisites.
No model architecture or training protocol was changed.

## Limits

This verifies fresh environments in the existing checkout, not a full remote
clone. The pinned reference source/checkpoint were already present; submodule
network checkout was inspected but not rerun, preserving the reference directory.
The isolated figure test deliberately excluded old local result artifacts.

Multi-hour full training commands were inspected against the CLI and checked
through bounded training, saving and evaluation paths, not rerun to completion.
Existing recorded full experiments remain the source of research results.
Smoke metrics are not forecasting results.

Linux, Windows, CUDA and other Python versions were not executed. Requirements
include version ranges/transitive dependencies, so future resolutions can differ.
The data are not in Git: the TA must obtain the trusted authors' CSI300 pickles.
External dataset mirrors, package servers, GitHub access and available hardware
cannot be guaranteed. Repeated commands need new output directories.
All README/code/input changes must be committed and pushed before the TA can use them.
