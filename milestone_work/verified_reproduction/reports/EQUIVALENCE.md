# Training equivalence

`verify_equivalence.py` compares an independently imported, unchanged copy of
the upstream architecture and the upstream `SequenceModel.train_epoch` against
our original runner's update mechanics. It uses identical weights, actual daily
CSI300 samples and dropout RNG state at widths 64, 128 and 256 on January 2,
2008; February 20, 2014; and March 24, 2020.

The nine cases match exactly for selected features, normalized labels,
predictions, MSE, raw/clipped gradients and the parameters after Adam updates.
Predeclared numerical tolerances are recorded in `equivalence.json` (1e-6 for
predictions/loss/gradients; 1e-7 for parameters). Exact observed zeros are
strong evidence of identical one-step behavior, not proof of full-run equality.

Documented differences:

- Reference DataLoader iterator creation advances PyTorch RNG, even without
  workers; our direct date loop does not. Same seed numbers alone do not produce
  identical full-run dropout trajectories. The controlled sweep retains the
  existing direct-loop convention consistently across widths/seeds.
- Our calendar filtering and five-date boundary purge change prediction rows
  relative to the unfiltered released loop, for documented compatibility and
  forward-label isolation reasons.
- Our preprocessing safely handles nonfinite labels and tiny cross-sections.
  On finite real CSI300 batches it is identical. On a 10-stock synthetic batch,
  the original `[0:-0]` slicing retains zero stocks; our fixed path retains ten.
- Original code saves only when training loss reaches its threshold. New
  bookkeeping adds atomic latest/best-validation/threshold checkpoints without
  changing loss, dropout, learning rate, optimizer or stop decision.
- CPU intra-op threads are pinned to two in the controlled sweep. Historical
  runs used a different/default threading setup; float32 reduction differences
  can produce small same-seed score differences over many updates. Exact checks
  above use identical threading on both sides.
- The authors' complete internal business implementation is unavailable.

No source reference files were edited. The snapshot copies match the reference
architecture and wrapper by SHA256 (checked during the audit). This comparison
uses the copied reference code; it never executes code under `original_paper/`.
