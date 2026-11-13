# Instructions for coding agents

## Project context

This course project extends MASTER with sparse inter-stock attention for stock-return forecasting. Read `README.md` first. Consult `project_proposal.pdf` for the submitted scope and `original_paper/original_paper.pdf` for the reference method before making research or architecture decisions.

The planned comparisons are dense MASTER, top-k attention neighbors, neighbors selected from trailing 60-day stock correlations, and k random neighbors. Planned datasets are CSI300, CSI800, and Yahoo Finance S&P 500 stock data. Do not describe planned work as implemented or claim unmeasured improvements.

## Maintain the README file inventory

- Update the `README.md` file inventory in the same change whenever you add, rename, delete, or materially repurpose a project file.
- Give each source, configuration, notebook, and documentation file a concise description of its actual purpose.
- Group large datasets, generated outputs, and checkpoints by directory; explain their contents and how they are obtained or regenerated. Do not enumerate every generated file or Git internal.
- Keep setup instructions, runnable commands, project status, and result links aligned with the repository. Clearly mark placeholders and unfinished work.

## Implementation practices

- Inspect existing code and conventions before editing. The authors’ reference implementation is an independent clone in `original_paper/MASTER/`; project-specific implementation and a dependency manifest have not been introduced yet. Document those choices when introduced.
- Keep the dense baseline available and make attention variants explicit through configuration.
- Preserve the supplied PDFs as reference documents.
- Treat everything under `original_paper/`, including the paper PDF and the cloned `MASTER/` repository, as read-only original source material. Do not edit, delete, rename, move, or update these files or the reference checkout unless the user explicitly requests it. Make project changes in separate files outside this directory.
- Record the source, revision, citation, and license of any imported research code.
- Keep credentials, downloaded datasets, caches, and large generated artifacts out of Git. Add appropriate ignore rules when those artifacts are introduced and document where they live.

## Experiment practices

- Use chronological train, validation, and test splits. Fit preprocessing on training data and construct features and neighbors only from information available at prediction time.
- Document the prediction target, feature definitions, stock universe, missing-data handling, and data provenance. For S&P 500 data, state how index membership and survivorship bias are handled.
- Record configuration, random seeds, dataset version or snapshot, split dates, and commands needed to reproduce each run.
- Compare methods using consistent data, tuning budgets, and backtest assumptions. Keep random-neighbor selection reproducible and document whether self-attention is included in k.
- Report IC, RankIC, ICIR, RankICIR, annualized excess return, and information ratio as appropriate. State metric definitions, aggregation across seeds, benchmark, transaction costs, and portfolio rules when results are added.
- Distinguish reproduced paper results, local baseline results, and extension results. Explain deviations from the paper or proposal.

## Verification and handoff

- For code changes, run relevant available checks and a small smoke test before expensive training. Do not start long or costly experiments unless requested or necessary within the authorized task.
- For documentation-only changes, verify file paths, links, and descriptions against the repository; no model tests are needed.
- Summarize what changed, what was checked, and any remaining limitations. Never invent successful runs or results.
