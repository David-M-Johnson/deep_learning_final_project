# Approved matched public-release test evaluation

The user explicitly authorized this phase after the nine training trials were
complete and checkpoint selection was frozen. This is separate from the original
validation-only sweep; its code/configuration and results remain unchanged.

## Fixed inputs before any test predictions

- Authors' supplied `csi300_opensource_0.pkl`, copied into sandbox/reference/model/.
- Our width-256 threshold-selected checkpoints at seeds 0, 1 and 2 (epochs
  26, 28 and 27). Validation-best checkpoints are deliberately not substituted.
- The one renewed CSI300 test pickle, with SHA256 pinned in
  configs/public_release_test.json, plus every model/code SHA256.
- Same CPU float32 architecture, gate beta 5, four temporal/two stock heads,
  and two CPU threads for every evaluation.
- Same intended 2020-07-01 through 2022-12-31 prediction segment. Actual final
  trading date is 2022-12-30. No final test dates are removed.
- The segment adapter exactly equals the stored legacy sampler range
  `[start_idx, end_idx)`; it does not change lookbacks or filter stocks by label.
  This addresses Qlib 0.9.7 exposing old pickle context dates, rather than
  silently including eight June context dates as test predictions.
- All 300 inference stocks on each of 611 dates are kept: 183,300 predictions.
  The label-only audit found 305 missing labels, leaving 182,995 scored rows.
  No day is entirely missing labels. No model predictions had been used to
  decide these rules.

## Unchanged reference evaluation

`paired_test_eval.py` imports the sandbox copy of the authors' original
`SequenceModel.predict` and uses it for every model. The only wrapper workaround
is setting its fitted flag to integer 0, avoiding the released loader's string
versus integer comparison bug. No reference source file is edited.

Original Pearson IC and Spearman RankIC are computed cross-sectionally each
day, then averaged. ICIR and RankICIR divide those means by population standard
deviations, without annualization. Missing labels are ignored by reference
pandas pairwise correlation; inference inputs remain intact. Independent daily
recomputation using the same `calc_ic` must agree with returned aggregates.
Any undefined date is an error under this fixed protocol, not silently dropped.

## Benchmark check and reporting

The author checkpoint is seed 0, so its comparison target is the renewed
train/test **seed-0** row, Sheet1 B14:E14 in performance.xlsx. The five-seed
mean in row 19 is contextual and must not be substituted for a seed-0 target.
Predeclared absolute tolerances: 1e-5 for IC/RankIC and 1e-4 for stability ratios.
Success/failure and signed differences are reported without tuning or changing
sample selection after observing the test results.

Report each of our three seeds independently, their mean/population standard
deviation, and the author's available seed-0 result. Only one released author
checkpoint is available for this universe/data version, so a paired five-seed
comparison is not possible from the files provided.

This matches **evaluation on the public release**, not unavailable corrected
proprietary original data, identical training, or portfolio/backtest assumptions.
No retraining, parameter search, new checkpoint selection, or AR/IR backtest is
performed. Test files are now explicitly evaluated, not still held out from us.
