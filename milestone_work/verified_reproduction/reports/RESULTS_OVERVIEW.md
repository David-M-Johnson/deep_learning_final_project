# Completed study overview

Nine trials: widths 64/128/256, seeds 0/1/2, fixed lr 1e-5, dropout 0.5,
full chronological Chinese training split, same stopping and validation-selection
rules. Widths 64/128 reached 40 epochs; width 256 reached the 0.95 training-loss
threshold at epochs 26/28/27. Latest, best-validation and threshold weights are
stored separately; best validation weights occur at epochs 3–6.

| Width | Best-IC-selected validation IC, mean ± std | RankIC at selected checkpoint, mean ± std |
| --- | ---: | ---: |
| 64 | 0.112166 ± 0.004088 | 0.111778 ± 0.001702 |
| 128 | 0.110716 ± 0.003718 | 0.112639 ± 0.005823 |
| 256 | 0.102843 ± 0.013252 | 0.103570 ± 0.009551 |

These validation-selected estimates are optimistic; three seeds do not establish
a statistically significant width winner. They are not test results.

The separate matched test comparison uses **only threshold-selected width-256
models** and the supplied authors' width-256 seed-0 checkpoint. Every model
uses 611 dates, the same 300 inference stocks each day, and the unchanged
reference predict/correlation code. Our three-seed mean test IC is
0.061261 ± 0.005297 and RankIC 0.068101 ± 0.003481; the supplied model's
corresponding values are 0.065416 and 0.069103. Workbook seed-0 values are
0.065333 and 0.068901. Agreement is close but outside the declared exact-match
tolerances; the discrepancy remains unresolved. No portfolio reproduction or
original proprietary-dataset replication is claimed.

The two report-ready illustrations are in `../../milestone_figures/`.
Detailed evidence is in the audit/equivalence/test reports and the copied
`../outputs/` tables. The current package does not evaluate width-64/128 test
performance or substitute their validation peaks for test results.
