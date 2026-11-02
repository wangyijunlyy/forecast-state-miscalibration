# Forecast-state granularity ablation

Protocol fixed on 2026-11-02 before evaluating the new partitions.

## Question and controlled factors

Separate the resolution of predictive mean/scale states from the addition of
forecast-horizon information. Sweep equal-frequency mean and log-scale bins
`b = 1, 2, 3, 4, 6, 8, 12, 16`, with one horizon group (respectively
1, 4, 9, 16, 36, 64, 144, 256 nominal states). Independently hold `b=4`
and vary equal-width horizon groups `g = 1, 2, 4, 8`
(16, 32, 64, 128 nominal states). In particular, 8x8x1 and 4x4x4 both
have 64 states but condition on different information.

For all eleven distinct partitions, use nested validation coordinate budgets
2,000, 10,000, and 50,000 (capped by the available reservoir), with the same
seed+60000 permutation as the existing budget controls. Fit boundaries on all
coordinates in each budget, then filter residual sources using raw ensemble
sample SD >1.01e-6. Retain the existing 1e-6 reconstruction floor, minimum
128 surviving sources per local pool, 20,000-order-statistic pool cap, and
100 midpoint-quantile reconstruction. Only partition resolution changes.

## Evidence population

Use all 249 full native checkpoints: seven backbones, 83 configurations,
training seeds 1, 2, 3. Use the saved validation fitting forecasts and the same
uniform test reservoir (up to 50,000 scalar coordinates) collected during
each complete native test traversal. Preserve actual horizon/channel IDs.
These are labelled retained-cohort ablations, separate from complete-test
primary results. No neural model is retrained or replaced. No test-based
selection changes the default partition or original primary results.

Report CRPS, QICE (percentage points), point MSE, and source occupancy:
eligible local pools, surviving counts, and test-coordinate global-fallback
percentage. Use a shared validation-fitted 4x4 diagnostic partition for all
coverage diagnostics within each task. Its choice does not affect CRPS/QICE.

Aggregate configurations equally within each training seed, then calculate
mean and sample SD over three seeds. Keep backbone score units separate.
Paired relative CRPS changes to 4x4x1 are computed within each task.

## Execution and validation

1. Add `granularity.py`: fit arbitrary bins with the existing empirical
   calibrator representation, retaining its exact reconstruction; test parity
   with portable DC16/DC64 and test the 128-source boundary.
2. Run `run_granularity.py` on 4090-wyj, reading original records without
   modifying them; write an isolated `granularity` result directory. Audit
   the shared DC16 budgets/DC64 against historical controls per task.
3. Independently audit all 249 task identities, 33 fits per task, arithmetic,
   budget nesting, coordinate pairing and fallback counts; export task,
   configuration and backbone CSVs with full precision.
4. Replace the original binary-choice subsection in place with resolution,
   horizon and budget evidence, preserving negative results; render tables
   from audited data, compile `ICDE/main.pdf` and check layout.

The frozen checkpoints and native sampling interfaces are those documented
in PROTOCOL.md and DEEPSTUQ_BACKBONE_PROTOCOL.md. This is an explicit
partition ablation of the full source-filtered calibration method.

## Table contract

The manuscript table uses CRPS without pooling backbone units. Panel (a)
reports eight mean/scale resolutions at the 50k cap; panel (b) reports smaller
DC16 budgets, three extra horizon partitions, and the existing unfiltered
source control. Mean and sample SD use three decimals. Full QICE, event,
point-error and fallback metrics remain in CSV/JSON records. This replaces
the earlier budget/source table without duplicating it. `render_granularity.py`
generates the table from audited summaries; values are not entered manually.

## Completed execution

All 249 tasks and 8,217 fits passed the server audit and the independent local
aggregation audit. Four historical score comparisons per task passed: DC16
at all three budgets and DC64 at the full budget (996 comparisons, each
checking CRPS, QICE, MSE, MAE, RMSE, and three Brier scores). The complete
18-test reliability suite passed in the server environment after providing
its historical `core.py` fixture in the relative directory expected by the
existing test. Local testing lacks PyTorch; server testing covers that path.

The results identify no universally optimal resolution. At the full budget,
36 mean/scale states beat DC16 on 53 configurations, lose on 21, and tie on
nine. Equal-count 8x8x1 beats 4x4x4 on seven backbone means and 52 configuration
means. Every 256-state fit at the 2k cap uses global fallback for every test
coordinate. These findings replace the earlier binary-choice argument
without retuning the fixed primary benchmark.
