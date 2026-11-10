# Mean anchoring of 36-state FSDC: Diffusion-TS experiment

Scope: nine Diffusion-TS datasets, three original training seeds, 27 frozen checkpoints. Keep ILI horizon 36 and all other horizons 192. Existing manuscript results remain unchanged. New variants are explicitly ablations.

Methods:

1. Native: unchanged native 100-member ensemble.
2. DC36: unchanged validation-fitted six-by-six empirical FSDC, loaded from the audited compiled record.
3. DC36MeanPreserving: subtract each actual 100-point template's arithmetic mean before reconstruction at the Native mean/scale.
4. DC36ControlledMean: the same centered template plus alpha times a separate state bias. State IDs use the original validation-fitted 6x6 boundaries; mean biases are not indexed by fallback-template ID.

The mean bias is fitted on the existing native TRAIN calibration reservoir. For state g with n>=128 coordinates, b_g=sum(y-mu)/(n+128); otherwise b_g=0. This uses raw forecast errors, with no division by predictive scale, and therefore retains all fitting coordinates. Quantile/source filtering remains unchanged in the distribution template.

Choose one alpha per task on the chronologically later VALIDATION reservoir: clip(sum((y-mu)*b_g)/sum(b_g^2),0,1), or zero when the denominator is zero. The original train/validation segment definitions supply chronological separation. Validation also fits the existing distribution template and served original native checkpoint selection; this experiment does not claim a new independent validation dataset. The point-bias fit never uses validation targets, and neither fit nor strength selection uses any test target. Fixed shrinkage/minimum count 128 is chosen before test scoring and is not tuned on the test results.

Using the existing calibration templates preserves the original FSDC fitting population and makes centering the only change in the mean-preserving ablation. The optional mean correction may still worsen held-out MSE under temporal shift. That is an experimental outcome, not an audit failure.

Evaluate a common retained test reservoir for direct cohort parity, then independently traverse every batch of the complete original native test iterator with seed=training_seed+30000. Keep Native, DC36, mean-preserving and controlled-mean forecasts paired at every coordinate. State/event metrics use the same training-fitted 36-state diagnostic and historical validation thresholds as the primary campaign. Compare Native and DC36 scores against the existing complete-test records.

Report CRPS, QICE in percentage points, mean-forecast MSE/MAE/RMSE, state coverage and Brier. Average all nine configurations within each seed, then mean/sample SD over the three run averages. Do not mix cohort and full-test scores. Record each fitted alpha, validation MSE, state counts and maximum output-mean deviation. No neural network is retrained and no additional backbones are included.

## Completed evidence

All 27 full-test tasks and paired retained cohorts completed without failures. Native/original-FSDC scalar metrics reproduce the primary campaign. Three numerical tests pass both locally and on the server. Independent reconstruction from raw training/validation caches verifies all fitted biases and strengths. Local auditing verifies state/event arithmetic, paired populations and mean-preserving MSE. The controlled-mean variant lowers backbone-average MSE by 6.520% and CRPS by 7.116% relative to Native, with both metrics improving in eight of nine three-seed dataset means. ExchangeRate worsens slightly in MSE and substantially in CRPS. Original FSDC retains the lowest aggregate CRPS among these four methods. Results are in `results/mean_anchor_diffusionts/REPORT.md`; the manuscript default is unchanged.
