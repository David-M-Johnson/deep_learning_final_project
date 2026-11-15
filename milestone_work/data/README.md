# Chinese MASTER data

Use the authors' renewed **opensource.zip**, containing CSI300 and CSI800
training, validation, and test Qlib sampler pickles. Sources from the pinned
upstream README:

- https://mega.nz/folder/MS8mUTbL#qeVz3KR1-MyXc_uLPtkvTg
- https://1drv.ms/f/c/652674690cc447e6/Eu8Kxv4xxTFMtDQqTW0IU0UB8rnpjACA5twMi8BA_PfbSA

From `milestone_work/base_model/` after environment setup:

```sh
../.venv/bin/python download_data.py --list
../.venv/bin/python download_data.py --name opensource.zip
# Alternative after a manual download:
../.venv/bin/python download_data.py --archive ../data/opensource.zip
```

Expected extracted files in `milestone_work/data/opensource/`:
`{csi300,csi800}_dl_{train,valid,test}.pkl`.
The archive, extracted data, and download manifest are ignored by Git. The
manifest records source, retrieval time, archive bytes and SHA256; training
records SHA256 of every split actually used. No publisher SHA256 was supplied.
Only load trusted upstream pickles; pickle deserialization executes Python.

Inputs: eight observations of Alpha158 stock factors and 63 Chinese index
market features. Label: `close[t+5]/close[t+1]-1`. Upstream feature processing:
training-fit median/MAD normalization, clipping to [-3,3], fill missing features
with zero. Original training labels are already filtered/standardized; renewed
open-source training labels need filtering and cross-sectional standardization.
The runner implements the latter, avoiding double processing when
`--data-version original` is explicitly selected.

The upstream README documents flaws in previously released original
validation/test data and a remaining missing-label/cross-section issue in
renewed validation. Downloading renewed data does not repair that issue; the
runner retains every supplied inference stock but cannot restore absent rows.
Thus renewed-data results must be distinguished from paper-data results.

Splits expected by the runner: train 2008-01-01..2020-03-31, validation
2020-04-01..2020-06-30, test 2020-07-01..2022-12-31. Actual split extents are
logged; only rows within the requested segment are trained/scored. The renewed
validation sampler starts March 20, 2020; those context rows are excluded from
validation scoring, which starts April 1. `audit.json` records all six local
file checksums, date ranges and first-batch shapes from implementation checks. By default the last five observed dates of train/validation
are conservatively purged to prevent labels crossing split boundaries.
Dataset construction and membership follow the authors' supplied Chinese
universes, not a newly reconstructed point-in-time universe. The original raw
source and preprocessing fit cannot be independently verified from pickles alone.
Data access does not imply the code's MIT license applies to financial data;
retain upstream attribution and observe the source provider's data terms.

## Local copy

All six extracted CSI300/CSI800 pickles were copied from
`david/reproduction/data/opensource/` without moving or changing the originals.
`copy_manifest.json` records SHA256 and size of every copied file; destination
checksums were compared against source checksums. `audit.json` and
`download_manifest.json` are copied upstream/local acquisition records and may
contain historical paths, not runtime dependencies. Raw archives, caches and
environments are not copied. Files under this directory other than this README
are ignored by Git; obtain them again with the documented downloader when
cloning elsewhere. Code MIT licensing does not grant redistribution rights to
financial data.
