# Four external calibration baselines on the ICDE forecast benchmark

## Scope and evidence

Run all seven frozen backbones, 83 configurations and 249 existing checkpoints
from the canonical 36-state manifest. Preserve each backbone's original data,
forecast length, normalization, full native architecture and complete test iterator.
No backbone training is performed. The four baselines answer whether FSPC improves
over classical parametric, learned PIT, neural distributional and forest-based
postprocessing. Component ablations remain a separate experiment/table.

## Shared data and scoring

Every method fits the identical saved validation reservoir (at most 50,000
coordinates from at most 100 native batches/windows). No additional subsampling.
The existing FSPC reference uses these validation templates plus its separately
recorded training-derived mean bias; this difference in historical fitting data
must remain explicit in any resulting manuscript comparison. Validation was also
used by the original backbone for checkpoint selection; do not claim independence.
No test target, test metric or test visualization selects any fitted parameter.
The calibration caches do not retain forecast-origin timestamps, so no new claim
of chronological validation subdivision is made. Fixed hyperparameters avoid an
unverifiable within-cache temporal tuning split.

Regenerate original complete native test ensembles with the original seed+30000.
All four methods share each ensemble. Use S=100 deterministic midpoint quantiles
for distribution outputs. Stream the complete original test iterator; retained
cohort scores cannot substitute for complete-test results. Record Native parity,
iterator batches/windows/coordinates and checkpoint identity. Reuse existing
FSPC/GlobalDC/PIT scores only when task identity and Native parity agree.

Scores: empirical-ensemble CRPS, QICE in percentage points, MSE, MAE, RMSE,
median MAE, pooled/state quantile coverage, Brier scores and event reliability.
Reuse the original training-derived 6x6 diagnostic partition and the original
validation-derived per-channel event thresholds. Within backbone: average
configurations within each training run, then mean/sample SD over three runs.
Cross-backbone average: paired Native-normalized backbone scores, equally
averaged within run, then three-run mean/sample SD. Do not pool absolute units.
Save fitting time, transform time, metric time, memory and serialized model size.

## Method specifications

### EMOS

Gaussian exchangeable-member EMOS/NGR: mean a+b*mu, variance c+d*sigma^2,
c,d>=0, a,b unrestricted. Fit by analytic Gaussian CRPS with the existing
four-parameter implementation, two initializations and recorded L-BFGS-B status.
The sample SD uses ddof=1 and the shared 1e-6 numerical floor. Existing
GaussianAffine4P fits may be reused after exact calibration-array parity checks;
record the original fitting path, coefficients and optimizer status. This is the
specified exchangeable Gaussian EMOS variant, not every member-specific EMOS model.
Source: Gneiting et al. (2005), doi:10.1175/MWR2904.1.

### Qian2025SoftCRPS

Use the complete published soft-weighted CRPS algorithm (Section 3.3), not a
binned or fixed-knot replacement. As in the original experiments, input CDFs
are Gaussian approximations using the native ensemble mean and sample SD.
All validation sources remain, including collapsed-scale forecasts. For a
Gaussian input family the quantile-gap term in the CRPS solution is separable:
d_ij=sigma_i*(Phi^-1(tau_(j+1))-Phi^-1(tau_j)). Thus its common probability-gap
factor cancels and the solution is exactly a weighted CDF of standardized errors,
with weights sigma_i times the Epanechnikov mean-similarity kernel. Bandwidth is
the distance to the L-th nearest predicted mean, L=floor(N/K).

Follow Section 4.1.2: choose K in 1..20 by the minimum calibration mean-squared PCE, averaging
over 3000 presampled Uniform(0,1) probability levels. Store all candidate scores.
Use a persistent moment tree to evaluate the exact weighted CDF/quantiles without
an N-by-N allocation. Verify it against direct kernel-weighted calculations.
Operate on standardized errors rather than numerically saturated Gaussian PIT
values: ordering is identical and finite extreme errors remain distinct. For
zero bandwidth, use the kernel's equal-mean limit. For narrow mean neighborhoods, directly sum the same nonnegative kernel weights
instead of subtracting global prefix moments; this avoids catastrophic floating-point
cancellation. If all nearest neighbors coincide on the kernel boundary, use the
continuous outward-bandwidth limit, whose common kernel factor cancels and leaves
the original scale weights. No residual trimming, finite
PIT knot grid, tail clipping or discretization of conditioning means is applied.
Source: Qian et al. (2025), doi:10.1080/00401706.2025.2464004.

### DistributionRecalibration2022

Implement Algorithm 2 and Section 6.1 of Kuleshov and Deshpande (ICML2022).
Inputs are native quantiles at 0.1,...,0.9 plus tau~Uniform(0,1). The recalibrator
has two fully connected hidden layers of 20 units, PReLU activations and dense
skip links. Fit the original uniform-tau pinball objective, with one sampled tau
per observation/minibatch update. Per-feature/target affine standardization uses
calibration data only. Fixed benchmark choices (the paper does not specify these):
Adam, learning rate 0.001, batch size 1024, 200 epochs, final epoch weights;
no early stopping or test-based tuning. This adapts the fitting budget to the
shared full reservoir rather than restricting this method to 500 observations.
Evaluate 100 midpoint levels and sort their predictions (monotone rearrangement)
to expose a valid empirical distribution; save raw quantile-crossing counts.
Source: https://proceedings.mlr.press/v162/kuleshov22a/kuleshov22a.pdf

### QRF2016

Full Meinshausen quantile regression forests applied as ensemble postprocessing,
following the method family of Taillardat et al. (MWR2016). Use the maintained
quantile-forest==1.4.2 implementation. Retain every bootstrap leaf observation
(max_samples_leaf=None), weight observations inversely by leaf size, aggregate
leaf distributions across trees before calculating quantiles, and do not use
quantiles of tree means or one sampled observation per leaf.

Predictors available in this benchmark: native mean, sample SD and nine deciles
(11 features). No unavailable meteorological predictors are invented. Fixed
settings: 1000 trees, minimum leaf size 5, max_features=floor(11/3)=3, bootstrap
sampling and two fitting CPU threads (four parallel query chunks for prediction); no depth truncation or calibration subsampling.
Quantile interpolation is linear. These predictor/settings adaptations are
recorded and are not claimed to reproduce the original weather datasets.
Source: Taillardat et al. (2016), doi:10.1175/MWR-D-15-0260.1;
Meinshausen (2006), https://jmlr.org/papers/v7/meinshausen06a.html
Implementation: https://github.com/zillow/quantile-forest

## Execution and result policy

Use a separate server directory and an isolated venv inheriting the existing
backbone runtime; install new packages there. Start one worker per free GPU,
with bounded CPU threads and per-task locks. Serialize shared-dataset preparation
because native SolarEnergy loading extracts an archive. Save each fitted model
before opening held-out targets; restart from those models without refitting.
Failures remain visible, no baseline is replaced or silently omitted. Check one
complete task before filling the remaining queue. Preliminary and complete
results are labelled separately. Existing manuscript tables stay unchanged until
complete results are available and audited.

## Runtime repair on 2026-11-09

The native diffusion interface returns a checkpoint directory; accept an existing,
nonempty directory as well as a file. The native loader and checkpoint identity
remain unchanged. Dense zero-padded QRF leaf arrays can require 412 GiB when
collapsed forecasts create large leaves. New fits use ragged bootstrap-leaf
storage with the same 1000-tree sklearn forest, all bootstrap multiplicities,
inverse-leaf weighting and the reference package's linear weighted-quantile
interpolation. Regression checks match both tree assignments and predictions
against quantile-forest1.4.2 (including a large repeated-feature group) at 1e-12.
Previously completed results and fitted models are retained. The controller now
records failed tasks while continuing independent pending tasks, rather than
stopping the entire queue after the first failure.
