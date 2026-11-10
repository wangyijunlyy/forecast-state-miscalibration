# Final FSPC factorial ablation and deployment costs

Requested on 2026-11-10 following the Section V review. Reuse all 249 frozen tasks (83 configurations, seven backbones, three runs), existing training/validation reservoirs and the identical retained test coordinates. No backbone training or new Native sampling. Diffusion-TS uses its corrected unclipped reservoirs.

## Factorial comparison

Freeze the completed `UnifiedMean` parameters before reading test targets. Use a single mean correction from the full FSPC fit in both arms that enable mean correction; use the same source-filtered centered residual templates in both arms that enable shape correction. This isolates application of the two components, without retuning an arm on test outcomes.

| Arm | Location | Zero-mean shape |
| --- | --- | --- |
| Native | Native ensemble mean | Native ensemble deviations |
| MeanOnly | FSPC blended affine-plus-state mean | Native ensemble deviations |
| ShapeOnly | Native ensemble mean | FSPC centered residual template, rescaled by Native spread |
| FSPC | FSPC blended affine-plus-state mean | FSPC centered residual template, rescaled by Native spread |

MeanOnly shifts every Native member by the full model's location adjustment. ShapeOnly removes both learned mean adjustments, including the training-bias reference. Score the four outputs on exactly the same coordinates and historical state/threshold definitions. Check Native and FSPC against existing retained-cohort records, and verify mean preservation, mean-only spread preservation and conditional mean/shape contrasts. Report MSE, MAE, RMSE, median MAE, CRPS, QICE, state coverage and Brier components. Store paired factorial interaction contrasts in addition to component scores. Do not apply author-supplied full-test score overlays to these measured cohorts.

## Complete fit and online cost

Time the complete existing `fit_unified` protocol, including chronological splitting/purging or short-validation fallback, template fitting/compilation, training-bias fit, validation reference-strength fit, affine-plus-state fits, blend selection and final refit. Cached Native generation and file I/O are outside the fit interval. Three fits on identical inputs must reproduce the frozen default model's predictions and parameters before timing results are accepted.

Fit replay preserves template and mean parameters within absolute 1e-9 / relative 1e-10 tolerance. Finite-ensemble location-blend CRPS is piecewise linear and its optimizer argument can be nonunique. The existing bounded scalar optimizer uses `xatol=1e-7`. Verify that replay reproduces the historical selection CRPS and reference objective within 1e-10, while satisfying the same reference-MSE constraint. Record weight differences and bound their prediction effect by the weight difference times the candidate/reference span, plus parameter tolerance. Do not require equality of optimizer arguments on a flat optimum. This replay check does not relax the bitwise comparison of reference versus compiled application of the same frozen model.

Benchmark the final full FSPC reference `UnifiedMean.predict`, an optimized deployment representation with precentered templates and frozen location parameters, full reconstruction followed by three event queries, direct queries from Native rows, and direct queries from cached Native summaries. Use strict exceedance and compare reconstructed values during binary search to preserve threshold ties. Save the complete deployment state, reload it, and verify outputs and all three event probabilities against the reference. Measure maximum value differences, bitwise equality and Brier parity explicitly; do not claim exact equality on tolerance checks alone.

Time three paired repetitions in randomized operation order after parity warm-up, recording every duration and the median. Include full reference/compiled state sizes, compilation time, fit time, per-coordinate online latency, and per-task speedup. Native generation cost is reused only when its exact workload and timing are identifiable; do not substitute the old component-only fit cost. Record hardware, software, CPU thread limits and campaign concurrency. Execute CPU-only using existing saved forecasts.

## Aggregation and manuscript

Average configurations within each run, then report three-run mean and sample SD. Compute speedup and interaction per task before aggregation. Publish only complete 249-task results after independent coverage, numeric, source-parity and serialization audits. Keep factorial accuracy, full fit/online cost and compiled parity in separate tables. Preserve the old filtering controls and immutable raw campaigns. Label retained-cohort measurements explicitly; they do not replace complete-test main-table records.
