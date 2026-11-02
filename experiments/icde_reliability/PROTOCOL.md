# ICDE forecast reliability experiments

Fixed before new test evaluation on 2026-11-02. The user subsequently changed the target from regular research to EAB. This track change does not alter the fixed data, methods, fitting budgets, or test evaluation protocol. The EAB deliverable requires the complete publicly accessible reproduction artifact.

## Scope and split

All 237 checkpoint tasks (79 backbone/dataset/horizon configurations, independent training seeds 1--3) from the completed reproduction are retained. Training and checkpoints do not change. The existing model-specific native iterator and 100-sample prediction interface are reused.

The full chronological validation iterator supplies a diagnostic of Native and the existing training-fitted DC16. At most the first 100 validation batches/windows supply a uniform 50,000-coordinate calibration reservoir. Thus the calibration time segment does not overlap backbone training, but is also the native checkpoint-selection validation segment. Source validity is original ensemble sample standard deviation >1.01e-6; reconstruction uses the unchanged 1e-6 floor. Four mean groups, four log-scale groups, local minimum 128 surviving sources, and local/global pool cap 20,000 are fixed. Boundaries use every retained calibration forecast, while residual sources alone are filtered.

## Paired full-test methods

Native; training-fitted source-filtered DC16 (historical calibration reused); validation-fitted source-filtered DC16; validation-fitted global residual distribution with the same source filter; global PIT recalibration; original conditional PIT with four horizon, scale and channel groups. All methods use the same native predictions, retained calibration coordinates and 100 output members. PIT uses the original 41-knot monotone warp implementation. The global and conditional PIT methods are implementations with explicitly stated binning, not claimed reproductions of unrelated GP/Beta calibration algorithms.

The complete native test iterator is scored without coordinate or target exclusions. Prediction sampling seed remains training seed +30000. Recompute Native and compare CRPS/MSE/QICE with historical scores; report nonzero deltas rather than assume equality.

## Diagnostics and event probabilities

Use a common diagnostic partition fixed by the training-fitted DC16 mean/scale boundaries for all methods. Record signed coverages at 0.1,0.5,0.9; report pooled and count-weighted statewise errors on exactly the same states with >=128 evaluation coordinates. Also record all counts and empty/sparse states. Do not infer a failure solely from increased absolute errors in tiny groups.

For each channel, obtain 0.8,0.9,0.95 target quantiles from a uniform reservoir of up to 4096 historical validation horizon-coordinate rows. These fixed thresholds define events Y>threshold. Forecast probabilities are the fraction of 100 ensemble members exceeding the threshold. Compute Brier scores and probability-bin reliability on all test coordinates. This evaluates single-coordinate exceedance probabilities, not joint trajectory or multivariate event guarantees.

## Budget and mechanism controls

Save a uniform 50,000-coordinate native test reservoir during the complete traversal. On that identical reservoir compare the full source-filtered DC16 algorithm fitted with nested calibration budgets 2000/10000/50000, DC64, and unfiltered DC16. This is a labelled calibration-budget/mechanism experiment, separate from complete-test primary scores. No early stopped models or reduced architectures are substituted.

Measure native generation and each postprocessor separately using synchronized CUDA timing where relevant; retain fitting time, calibration state size and peak process memory. Record actual workload sizes. Aggregate within configuration across seeds; summarize each backbone on its own fixed tasks, and use common dataset/horizon subsets for direct cross-backbone comparisons. Raw statistics retain full precision; manuscript table standard deviations use actual sample SD rounded to three decimals.

## Completion evidence

Only valid results for all 237 task identities, all six paired full-test methods, complete iterator counts, method/split identities and independent numerical checks establish experiment completion. Existing result files must be matched to this protocol. Process liveness is verified from actual process command lines; a stale status file is insufficient.
