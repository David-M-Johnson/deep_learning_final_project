# Dense MASTER first: experiment plan

Updated November 14, 2026. This is a plan, not a report of completed experiments. Start with [remote computing instructions](remote_computing_instructions.md). No project training runner, dependency lock, S&P pipeline, or results have been implemented yet. The setup guide supplies bounded, copyable smoke-test examples using the read-only reference model.

## Sequence and success criteria

Follow these stages in order. Keep dense attention throughout stages 0-4; do not implement attention extensions until the dense U.S. baseline is established. A reduced-dimension run is a workflow check, not a paper reproduction.

| Stage | Work | Evidence required before proceeding |
| --- | --- | --- |
| 0. Remote workflow | Provision one NVIDIA GPU; connect by SSH; create one venv; run width-64 MASTER on synthetic inputs. | CUDA forward/backward succeeds; parameters update; checkpoint reload reproduces predictions; logs and environment record are saved; VM is stopped. Synthetic loss is not financial evidence. |
| 1. Small Chinese-market baseline | Dense CSI300, width 64, all 158 stock and 63 market features, 8-day input window. Start with five complete training dates and five validation dates, then one bounded epoch if needed. | Real data load; finite loss/gradients; dated predictions and checkpoint reload; measured runtime and peak GPU/RAM use; data snapshot and preprocessing verified. No requirement for strong IC. |
| 2. Reference-size Chinese baseline | Dense width 256; CSI300 first, then CSI800. Use full chronological splits and a documented reproduction protocol. | Five seeds, paper/reference comparison with deviations explained, prediction metrics and eventual portfolio evaluation. Investigate discrepancies; do not declare a match based on width alone. |
| 3. Small U.S. baseline | Build and audit S&P data pipeline, then train dense width 64 from scratch. First bounded dates, then a short training run. | Verified features/targets, temporal alignment, membership policy, missing-data policy, training-only preprocessing, U.S. gate inputs; finite training, validation predictions, save/reload. |
| 4. Reference-size U.S. baseline | Dense width 256 on the frozen U.S. dataset; train from scratch across multiple seeds. | Full training/evaluation with recorded stopping rule, runtime, metrics and backtest assumptions. Establish a reproducible U.S. baseline, regardless of whether accuracy resembles Chinese results. |
| 5. Attention extensions | Dense vs. top-k attention vs. trailing-60-day correlation vs. reproducible random neighbors. | Controlled comparisons using identical data, model width, seeds, tuning budgets and portfolio assumptions; report predictive quality, runtime and memory. |

Do not use a width-64 checkpoint as initialization for the width-256 baseline. Their tensor shapes differ, and training from scratch keeps the experiment interpretable. Reuse formulas and code, not fitted Chinese-market preprocessing or model weights, for the U.S. stages.

## Baseline specification

The [paper](../original_paper/original_paper.pdf), experiment section, specifies Alpha158 inputs, lookback 8, prediction interval 5, hidden dimension 256, learning rate 1e-5, four temporal heads, two stock heads, beta 5 for CSI300 and 2 for CSI800, at most 40 epochs, and five random initializations. The reference [main.py](../original_paper/MASTER/main.py) uses dropout 0.5. Width 64 is our proposed workflow setting; retain the head counts, dropout and dense attention initially.

| Setting | Workflow check | Chinese reference-size evaluation |
| --- | --- | --- |
| Hidden width | 64 | 256 |
| Features | 158 stock + 63 market | Same |
| Lookback | 8 trading observations | Same |
| Temporal / stock heads | 4 / 2 | Same |
| Learning rate | 1e-5 | 1e-5 initially; log any tuning |
| Beta | 5 for CSI300 | 5 CSI300 / 2 CSI800 |
| Seeds | 0 | 0, 1, 2, 3, 4 |
| Training budget | Five training dates, then a bounded epoch | At most 40 epochs under the chosen reproduction stopping protocol |
| Test use | None | Once protocol and tuning are frozen |

Keep all eligible stocks for a selected date together: the model attends across that day's stocks. Reduce the number of dates for a smoke test rather than randomly minibatching stocks. Record any training-label filtering that changes the daily cross-section.

The paper's target is `close[t+5] / close[t+1] - 1`, also expressed as `Ref($close,-5)/Ref($close,-1)-1` in the supplied Qlib YAML. This is not `close[t+5]/close[t]-1`. Features end at t; implement and test the exact trading-day alignment. For backtests, declare when information becomes available and when trades execute. Purge training/validation samples whose forward labels cross into the next split; record the resulting boundary exclusions and any deviation from the released samples.

Paper split: training Q1 2008-Q1 2020, validation Q2 2020, test Q3 2020-Q4 2022. The supplied YAML spells these out as 2008-01-01 to 2020-03-31, 2020-04-01 to 2020-06-30, and 2020-07-01 to 2022-12-31. Audit actual dates in downloaded objects rather than assuming they match this configuration.

## Reproduction limits to resolve at stage 2

The authors' [README](../original_paper/MASTER/README.md) says their full original business codebase is confidential and documents errors in published validation/test preprocessing and an additional validation issue noted June 26, 2025. Renewed open-source data differ from original data. Therefore:

- Record whether each split is original or renewed open-source data, download date, source URL, checksum, actual date range, feature columns and universe.
- Compare renewed-data runs with the corresponding supplied results when available. Distinguish original paper results, supplied checkpoint evaluation and locally trained results.
- Original training labels already received extreme-label filtering and cross-sectional normalization. Open-source data require the extra processing described in `base_model.py`; do not process twice. At inference retain the eligible stock cross-section and mask invalid labels only for scoring.
- The released loop stops at a training-loss threshold (example 0.95) and saves only upon crossing it. It prints validation metrics but does not select the best validation checkpoint. Our final runner must save a last checkpoint even when the threshold is never reached. Record the selected checkpoint and stopping reason.
- A validation-patience rule would be a separate experimental choice. My earlier suggestion of patience 3 was a development suggestion, not an existing reference-code behavior or verified paper reproduction protocol.
- Review supplementary details and data notices before deciding what constitutes an acceptable reproduction. Explain gaps; do not repeatedly tune against the test set to force a match.

## U.S. data contract before stage 3

Write a data manifest before training. Yahoo Finance is the proposed source, not an already working pipeline.

1. **Universe:** Prefer point-in-time historical S&P 500 membership, including removals and ticker changes, with membership effective dates. If only today's constituents are available, label the experiment a fixed-current-constituent prototype with survivorship bias. A full training run on that prototype does not remove this limitation.
2. **Stock inputs:** Recreate the actual Alpha158 expressions in a fixed order, not 158 arbitrarily chosen indicators. Record required raw fields, price/split/dividend and volume adjustments, warmup history, time zones, exchange calendar, units, and formula tests. Indicators may require more history than the model's 8-day input window.
3. **Target:** Use the same forward-return formula above, and document any adjustment differences. Never put future returns among the input features.
4. **Market gate:** Select and document three U.S. market representations. S&P 500 / S&P MidCap 400 / their combined S&P 900 is a candidate mapping, subject to data availability. Confirm actual return and trading-amount expressions against the reference feature columns. The reference Qlib example uses index returns and normalized amount statistics; the prose summary of “index values” is not a sufficient implementation specification. ETF turnover is not index constituent dollar turnover. Any proxy or changed formula must be explicit and identical across comparisons.
5. **Preprocessing:** Fit robust normalization on training only, then apply frozen values to validation/test. Match clipping and missing-value handling; do not forward/backfill using future data. Preserve valid inference stocks even if labels are missing. Define treatment of IPOs, delistings, holidays, missing bars, and stocks without enough history.
6. **Splits:** Select chronological U.S. dates based on verified data coverage before model tuning. Keep enough warmup history, purge overlapping forward labels at boundaries, and reserve the test period. U.S. split dates are intentionally not yet fixed.
7. **Provenance:** Save download arguments, package versions, retrieval time, membership source, raw snapshot identifiers/checksums, processed-data version and licensing/access constraints. Cache snapshots rather than silently redownloading different data for each seed.

## Lean implementation layout (proposed, not present yet)

Use one project checkout and one `.venv` per machine. Keep the same environment for small and large widths; width is configuration, not a reason for another venv. Use plain Python/PyTorch first, Qlib only where needed for published pickles/features/backtests. Avoid Docker, notebooks as the only runner, distributed training, mixed precision and compilation until the float32 workflow is correct and timed.

```text
~/work/deep_learning_proj/       # VM checkout; this Mac checkout remains separate
  .venv/                        # VM-local packages, never copied from Mac
  david/                        # these guides
  original_paper/                # immutable reference material
  src/                          # future project model / data / training modules
  configs/                      # future explicit experiment settings
  requirements/                 # future curated dependencies and validated locks
  tests/                        # future feature, split and checkpoint checks
~/master_data/                  # ignored/external downloaded and processed snapshots
~/master_runs/<run-id>/         # external logs, manifests, predictions, checkpoints
```

When introducing these future project files, update the README inventory and add ignore rules for environments, caches, data and outputs in the same change. Keep reusable code outside `original_paper/`. If importing research code, preserve MIT license and attribution: SJTU-DMTai/MASTER, revision `de8f58557096abde4216a701b35fc4368158d111`, Li et al., AAAI 2024. The smoke guide imports this pinned model read-only; it does not vendor or alter it.

Implement one configurable runner after the workflow check: explicit device selection with CUDA failure if requested but unavailable, pathlib paths, bounded training-date mode, seed handling, training/inference label policies, last/selected checkpoints, resume state and evaluation. Resume checkpoints should include optimizer, epoch and RNG state; a model-only smoke checkpoint proves inference reload, not faithful training resumption.

## Run records and evaluation

Each run should save: Git revision and dirty status, upstream revision, full configuration, seed, command, Python/package lock, GPU/driver/CUDA versions, dataset manifest/checksums, actual split dates, preprocessing fit interval, target, eligible stock counts, checkpoint selection/stopping reason, wall time and peak GPU memory. Also measure host RAM; large Qlib pickle objects can exhaust RAM before GPU memory is full.

Compute daily cross-sectional Pearson IC and Spearman RankIC, then average over dates. Define ICIR/RankICIR explicitly: the lightweight reference uses mean daily correlation divided by population standard deviation, without annualization. Record undefined/constant-label days rather than silently dropping them without counts. Report mean and standard deviation across seeds, retaining per-seed results.

Add annualized excess return and information ratio only after a verified portfolio backtest. Specify benchmark, rebalancing, portfolio selection/weighting, execution price and delay, transaction costs, turnover and annualization. Synthetic or five-date checks need no portfolio claims. Use one common protocol for dense and sparse variants.

## Attention work begins only after stage 4

Freeze dense baselines and data. Make attention mode and k explicit configuration; define whether self is included in k. Select correlation neighbors using only trailing 60 available trading days, with missing-history and tie rules. Seed random neighbors and declare whether graphs are fixed or change by date. Top-k computed from all dense scores still pays the pairwise score cost; a dense mask does not demonstrate sparse computational savings. Measure actual implementation time and memory alongside accuracy.

## Verification status

These documents were checked against local reference code, the proposal and the paper, and linked official cloud/PyTorch documentation. No VM was provisioned, no billing settings were changed, no dependency environment was installed, and no GPU or financial-data experiment was executed during documentation authoring. Dependency/pickle compatibility, available credits/quota/capacity, runtime and reproduction accuracy remain to be verified on the VM.
