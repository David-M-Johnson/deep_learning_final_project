# Improving Transformer Models for Forecasting

Deep learning course project extending **MASTER: Market-Guided Stock Transformer for Stock Price Forecasting** with sparse inter-stock attention.

**Team:** Brian Yin, David Johnson, and Marvin Widjaja.

**Status:** Project setup and documentation. The authors’ reference code is cloned under `original_paper/MASTER/`. Project-specific model changes, data pipelines, and experiments have not been added yet.

## Project goal

MASTER uses market-guided feature selection and alternates intra-stock and inter-stock attention to forecast stock returns. Our project investigates whether limiting inter-stock attention to a small set of relevant neighbors improves forecasting performance.

The submitted [project proposal](project_proposal.pdf) defines the planned scope:

1. Reproduce the original dense-attention MASTER baseline on CSI300 and CSI800.
2. Implement sparse inter-stock attention using either the top-k attention scores or neighbors selected by stock correlations over the past 60 days.
3. Compare these methods with dense attention and a control using k randomly selected stocks.
4. Build a data pipeline for S&P 500 stocks from Yahoo Finance and evaluate the methods on that market as well.

Sparse attention is a hypothesis to test; no improvements have been demonstrated in this repository yet.

## Evaluation plan

Evaluate across several random seeds using the same data splits and evaluation settings for each method:

| Category | Metrics |
| --- | --- |
| Prediction quality | IC and RankIC |
| Prediction stability | ICIR and RankICIR |
| Portfolio performance | Annualized excess return and information ratio from Qlib backtests |

Record data sources, date ranges, preprocessing, split boundaries, model settings, neighbor-selection rules, k values, seeds, and backtest assumptions alongside each experiment. Compute correlation-based neighbors from information available at prediction time to prevent future-data leakage.

## Getting started

The expected implementation stack is Python, PyTorch, and Qlib. Dependencies, supported versions, dataset setup, and training commands are still to be defined. There is no runnable training or evaluation workflow yet.

Next steps:

- [ ] Establish the environment and dependency manifest.
- [ ] Add and document the baseline implementation and its provenance.
- [ ] Prepare CSI300 and CSI800 data and chronological splits.
- [ ] Reproduce the dense baseline.
- [ ] Add sparse attention variants and the random-neighbor control.
- [ ] Prepare the Yahoo Finance S&P 500 pipeline.
- [ ] Run repeated-seed evaluations and summarize results.

## File inventory

Keep this inventory current whenever project files are added, renamed, removed, or change purpose. List individual source and documentation files; group generated outputs or large data collections by directory and explain their contents. Git internals are excluded.

| Path | Purpose |
| --- | --- |
| [README.md](README.md) | Project overview, scope, setup status, evaluation plan, and maintained file inventory. |
| [AGENTS.md](AGENTS.md) | Instructions for coding agents, including documentation and experiment practices. |
| [original_paper.pdf](original_paper/original_paper.pdf) | Reference paper: MASTER: Market-Guided Stock Transformer for Stock Price Forecasting, the baseline this project extends. |
| [original_paper/MASTER/](original_paper/MASTER/) | Independent clone of the authors’ official repository; includes model code, data documentation, supplied checkpoints, Qlib integration, and supplementary materials. See its README for the upstream file details. |
| [project_proposal.pdf](project_proposal.pdf) | Proposal submitted to the professor; describes the planned sparse-attention extensions, datasets, comparisons, and metrics. |

## References

- Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, and Sen Huang. *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*. AAAI 2024. Local copy: [original_paper.pdf](original_paper/original_paper.pdf). Paper identifier: arXiv:2312.15235.
- [Submitted project proposal](project_proposal.pdf).

## Reference code provenance

- Source: [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER).
- Local location: `original_paper/MASTER/` (an independent Git clone).
- Revision cloned: `de8f58557096abde4216a701b35fc4368158d111`.
- License: [MIT](original_paper/MASTER/LICENSE); retain the upstream license when reusing code.
- Associated paper: MASTER, cited above. The public repository provides core code and processed data; the authors’ complete internal experiment codebase is confidential.

To recreate the reference checkout:

```bash
git clone https://github.com/SJTU-DMTai/MASTER.git original_paper/MASTER
git -C original_paper/MASTER checkout de8f58557096abde4216a701b35fc4368158d111
```

## Results

No experiments have been run or results recorded yet. Add reproducible commands, configurations, and links to result artifacts here as they become available.
