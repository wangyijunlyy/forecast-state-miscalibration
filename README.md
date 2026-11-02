# Forecast-State Miscalibration

Code, data, configurations and experiment records for **Forecast-State Miscalibration in Probabilistic Time-Series Analytics**. Neural model weights are not distributed; the native training code creates checkpoints locally.

## Experiments

- Seven backbones,83 dataset–horizon configurations,and three training seeds:249 training runs.
- Empirical residual calibration,global/conditional PIT,and two-/four-parameter Gaussian corrections on paired complete native validation/test forecasts.
- Source-validity,budget,state-assignment,and exact template-compilation controls.
- DeepSTUQ (**ICDE2023**) is the seventh backbone on PEMS03/04/07/08. Its twelve native training replications receive the same primary corrections,Gaussian controls,training DC16/DC64 comparison,budget and source-validity ablations,state-assignment controls,and compilation benchmarks as the other six backbones.

Backbone summaries first average configurations within each seed,then compute sample SD across three training seeds. Manuscript values use three decimal places;raw records retain full precision.

## Download

Download the code and records from [v1.1.0](https://github.com/wangyijunlyy/forecast-state-miscalibration/releases/tag/v1.1.0). The unchanged dataset parts are available in [v1.0.0](https://github.com/wangyijunlyy/forecast-state-miscalibration/releases/tag/v1.0.0):

| Files | Contents |
|---|---|
| `code-and-records.tar.gz` | Code,configurations,reference records,source-restoration patches,dependency locks,analysis scripts,and measured figure data |
| `data-only.tar.gz.part-000` through `part-009` | Original dataset files in the native loader layouts;no model checkpoints |
| JSON manifests | File names,byte sizes and transfer-integrity records |

The dataset archive is approximately639MB compressed. Model checkpoint files (`.pt`, `.pth`, `.ckpt`, `.safetensors`) are excluded from both archives.

## Rebuild tables and figures

With Python3.10–3.12,download the code archive and run:

```sh
mkdir artifact
tar -xzf code-and-records.tar.gz -C artifact
python -m pip install numpy==1.26.4 scipy==1.15.3 matplotlib==3.10.3
cd artifact
python experiments/icde_reliability/rebuild_analysis.py --workspace "$PWD"
python experiments/icde_reliability/audit_table_sd.py --results analysis_summaries --paper ICDE
```

These commands audit the recorded suites and regenerate the shared seven-backbone result tables and two quantitative figures. The SD auditor independently verifies all469 mean/SD cells against raw three-seed scores. No GPU,dataset download or neural model weights are needed for this analysis.

## Train and evaluate native models

Download all ten data parts,concatenate in their three-digit suffix order,and extract into the same artifact directory:

```sh
cat data-only.tar.gz.part-[0-9][0-9][0-9] > data-only.tar.gz
tar -xzf data-only.tar.gz -C artifact
```

Follow [the training/replay instructions](docs/ARTIFACT_REPLAY.md) to restore505 upstream source files,install the measured environment,relocate paths,and invoke `from_scratch_eab.py`. Native training uses Linux,Python3.10,PyTorch2.4.1+cu121,and a compatible NVIDIA GPU. The entry point uses the recorded native training settings,refits the calibration states locally,and performs complete native validation/test evaluation.

NsDiff on ETTh1 requires shared F/G initializers. The helper trains those locally from their actual saved configurations;no downloaded pretrained F/G parameters are required. DeepSTUQ's from-scratch route retains its full architecture,pretraining,all twenty AWA epochs,ten averaged snapshots,ten dropout passes,and temperature calibration.

## Evidence and interpretation

All249 primary,Gaussian,state-assignment and compiled-deployment task records are included. The twelve original DeepSTUQ training records also preserve the complete native method and provide a native-output parity reference. Fresh extraction of the code archive regenerates the shared result tables and both figures. The original private-checkpoint replay in a clean dependency environment reproduced321 native metric comparisons exactly. The public release distributes the recorded evidence and from-scratch training route;it does not distribute those checkpoints.

Traffic scores retain native flow units;the other six backbones use model-normalized units. Each correction is compared within the backbone's matched task set,and the shared tables do not pool these scales into an absolute ranking.

## Attribution

Public upstream bases plus measured patches reconstruct the native model sources. Source identities,available licenses,and modifications are preserved in package provenance. Upstream repositories and datasets retain their original terms;this repository does not relicense them. The manuscript PDF and author metadata are separate from this artifact release.
