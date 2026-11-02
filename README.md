# Forecast-State Miscalibration

Reproducibility artifacts for **Forecast-State Miscalibration in Probabilistic Time-Series Analytics**.

This study examines probability errors within observable predictive mean–scale states, residual amplification caused by nearly collapsed fitting scales, temporal fitting effects, and probability-delivery cost. Calibration families are compared on paired native forecasts rather than independently sampled predictions.

## Experiment scope

- Six long-horizon backbones,79 dataset–horizon configurations,and three training seeds:237 native checkpoints.
- Empirical residual calibration,global/conditional PIT,and two-/four-parameter Gaussian calibration on complete native validation/test iterators.
- Source-validity,budget,state-assignment,and exact template-compilation controls.
- DeepSTUQ (**ICDE2023**) on PEMS03/04/07/08 with three training seeds:12 full native training replications.

Reported uncertainty is the sample standard deviation across three training seeds. Backbone summaries first average configurations within each seed. Manuscript values use three decimal places; raw records retain full precision.

## Download the complete artifacts

The [v1.0.0 release](https://github.com/wangyijunlyy/forecast-state-miscalibration/releases/tag/v1.0.0) provides:

| Files | Contents |
|---|---|
| `code-and-records.tar.gz` | Actual code/configurations,reference records,dependency locks,source-restoration patches,analysis scripts,and measured figure data |
| `base-assets.tar.gz.part-00` through `part-05` | Native data,237 checkpoints,conditional predictors,shared initializations,and training-fitted residual states |
| `traffic-assets.tar.gz` | Twelve fully trained DeepSTUQ models and associated model dependencies |
| JSON manifests | Ordered filenames and exact byte sizes |

### Rebuild tables and figures without a GPU

Download and extract `code-and-records.tar.gz`,then run:

```sh
mkdir artifact
tar -xzf code-and-records.tar.gz -C artifact
python -m pip install -r requirements-analysis.txt
cd artifact
python experiments/icde_reliability/rebuild_analysis.py --workspace "$PWD"
python experiments/icde_reliability/audit_table_sd.py --results analysis_summaries --paper ICDE
```

The first command audits all five experiment suites and regenerates the result tables and two quantitative figures. The second independently checks all442 mean/SD cells against raw three-seed scores. These analysis commands do not require the large model/data archives.

### Recompute native model forecasts

Download all six base parts and the traffic archive. Concatenate the parts and extract into the same artifact directory:

```sh
cat base-assets.tar.gz.part-* > base-assets.tar.gz
tar -xzf base-assets.tar.gz -C artifact
tar -xzf traffic-assets.tar.gz -C artifact
```

Follow [ARTIFACT_REPLAY.md](docs/ARTIFACT_REPLAY.md) inside the extracted package to restore505 upstream source files,install the pinned environment,relocate paths,and recompute complete native validation/test forecasts. Native replay uses Linux,Python3.10,PyTorch2.4.1+cu121,and a compatible NVIDIA GPU. Instructions cover supplied-checkpoint replay,residual-pool regeneration,from-scratch training,Gaussian controls,and the full DeepSTUQ method.

## Evidence and interpretation

All237 primary,Gaussian,state-assignment,and compiled-deployment task records and all12 traffic task records pass their independent audits. A fresh dependency environment and public-source restoration reproduce321 distinct native metric comparisons exactly across the six long-horizon backbones and a complete traffic task. Exact compilation preserves reconstructed values and threshold-event probabilities bitwise.

The results include tradeoffs:Gaussian2P obtains lower validation-fit average CRPS than the binned correction on five long-horizon backbones;the binned correction has smaller state coverage error than both Gaussian controls on four. Traffic and long-horizon scores use different matched task sets and are not combined into an absolute ranking.

## Sources and attribution

The code/records package preserves source identities,available licenses,and the measured modifications. Public upstream bases plus patches reconstruct the unbundled model-source files. Upstream repositories retain their own terms;this repository does not relicense third-party implementations or datasets. See package provenance and [the DeepSTUQ protocol](docs/DEEPSTUQ_PROTOCOL.md) for the original ICDE2023 source revision and integration details.

The manuscript PDF and author metadata are separate from this reproducibility release.
