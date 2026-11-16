# Two milestone figures

Exactly two 400-dpi PNGs are supplied. Text uses the actual Computer Modern
Roman font bundled with Matplotlib (`cmr10.ttf`), with Computer Modern math.
Every panel has four visible 0.75-point borders, restrained colors and only
three/four light horizontal grid lines. Points are individual measurements;
horizontal black marks are arithmetic means, not confidence intervals.

## 01_three_seed_validation.png

**Caption:** Dense MASTER validation performance for hidden widths 64, 128 and
256 across three training seeds. Each seed's checkpoint is selected by highest
validation IC, with the same dataset and training budget at every width. The
left-to-right point offsets within each width denote seeds 0, 1 and 2. Black
marks show three-seed means. Widths 64 and 128 show similar validation averages;
width 256 varies more across these seeds. These are selected validation scores,
not held-out test results, and do not establish a significant width winner.

## 02_matched_test_reproduction.png

**Caption:** Matched public-release evaluation on CSI300 at hidden width 256.
The authors' supplied seed-0 checkpoint and our three threshold-selected models
use the same 611 test dates, 300 inference stocks per date, target labels and
unchanged reference evaluation code. Open squares show reported workbook seed-0
values; grey diamonds show our evaluation of the supplied checkpoint. Green
points show our seeds 0, 1 and 2; black marks show their mean. Test performance
is comparable in scale, but the supplied checkpoint does not exactly reproduce
its workbook values. The residual difference remains unresolved. This does not
claim original proprietary-data or portfolio reproduction.

## Regenerate

After [workspace environment setup](../README.md), from `milestone_work/`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python milestone_figures/make_figures.py
```

The script reads only the copied completed-study records under
`verified_reproduction/outputs/`, not live sandbox files. It does not train,
select checkpoints or reevaluate test data. No external LaTeX installation is
required. The exact input hashes, plotted values, font and border settings are
recorded in `figure_sources.json`.

| File | Purpose |
| --- | --- |
| `README.md` | Captions, interpretation, styling and regeneration command. |
| `make_figures.py` | Deterministic two-figure builder using the copied result JSONs. |
| `figure_sources.json` | Exact plotted values, source hashes and styling provenance. |
| `01_three_seed_validation.png` | Nine seed observations and mean validation correlations by width. |
| `02_matched_test_reproduction.png` | Matched test results for the supplied reference and our width-256 seeds. |

The two compact figures and small provenance file are suitable for Git. Large
model/data/prediction artifacts remain ignored elsewhere in the workspace.
