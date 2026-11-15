# Baseline figures

From `milestone_work/base_model/`, after workspace environment setup:

```sh
# Compare all three copied widths:
../.venv/bin/python plot_epochs.py
# Publication figures for each copied width:
../.venv/bin/python plot_publication.py --width 64
../.venv/bin/python plot_publication.py --width 128
../.venv/bin/python plot_publication.py --width 256
```

Outputs are generated from `saved_runs/`, not the original live histories.
`comparison/` contains training MSE and validation IC/RankIC PNGs with an exact
source-value manifest. `width64/`, `width128/` and `width256/` contain vector PDFs,
300-dpi PNGs, history snapshots, captions and SHA256 source manifests.
Only this README is tracked by default; all generated figures are ignored.
For new local training histories, specify the run explicitly:

```sh
../.venv/bin/python plot_publication.py --run runs/csi300-width128 --output figures/width128-new
../.venv/bin/python plot_epochs.py runs/csi300-width64 runs/csi300-width128 runs/csi300-width256
```

Dots mark the best observed validation values, not necessarily saved checkpoints.
The number of displayed epochs may differ between model snapshots. Do not treat
single-seed validation plots as independent test results or infer uncertainty
bands. Falling training loss and declining validation scores are consistent with
overfitting; curves alone do not establish its cause.

`plot.py` additionally visualizes daily prediction diagnostics after evaluating
a copied checkpoint (prediction CSVs are regenerated to avoid mixing epochs):

```sh
../.venv/bin/python plot.py runs/reload-validation-width128 --split valid \
  --history saved_runs/csi300-width128/seed-0/history.csv \
  --output figures/width128-reload-diagnostics
```

For an evaluation directory without `history.csv`, `plot.py` expects history in
its parent. Instead pass `--history` pointing to the matching snapshot history
when evaluation and snapshot directories are separate.
