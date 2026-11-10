# Forecast-State Miscalibration

Experiment code for **Forecast-State Miscalibration in Probabilistic Time-Series Analytics**. The current method is forecast-state probability calibration (**FSPC**): source-filtered residual templates in 6 × 6 forecast-mean/log-scale states, centered reconstruction, and a validation-selected affine-plus-state mean blended with a training-bias reference.

This branch contains code, task configurations, dependency specifications and reproduction instructions. Datasets, trained weights, cached predictions, fitted calibrators, experiment results and manuscript outputs are generated or downloaded locally; they are not checked in.

## Quick check without a GPU

Use Python 3.10 (the native experiments used 3.10.19):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/icde_reliability -p 'test_final_fspc.py' -v
python -m unittest discover -s experiments/icde_reliability -p 'test_fspc_unified.py' -v
python scripts/reproduce.py list
```

The tests use generated arrays to check isolation of the mean/shape arms, complete FSPC fitting, strict event-threshold ties, bitwise compiled/reference equality and saved-state reloads. Generated arrays are numerical tests, not benchmark results.

## Reproduce a native-model experiment

Native training/inference requires Linux, an NVIDIA GPU, and the measured PyTorch 2.4.1 + CUDA 12.1 environment. RTX 4090 GPUs were used for the original experiments. Complete FSPC CPU timings used Intel Xeon Gold 6426Y processors, two CPU threads per worker, and four workers.

```sh
python scripts/setup_environment.py --requirements requirements-native-lock.txt
python -m pip install -r requirements-external.txt

# Restore pinned public sources and apply the experiment's source patches.
python scripts/reproduce.py prepare --workspace "$PWD/work"
# Download only the dataset archive, in the layouts expected by native loaders.
python scripts/reproduce.py fetch-data --workspace "$PWD/work"

# A single configuration; train once, reuse its checkpoint for every correction.
python scripts/reproduce.py train --workspace "$PWD/work" \
  --task K2VAE_ettm1_h192_seed1 --gpu 0
python scripts/reproduce.py evaluate --workspace "$PWD/work" \
  --task K2VAE_ettm1_h192_seed1 --gpu 0 --controls --external
```

`configs/tasks.json` fixes all **249 runs / 83 configurations / seven backbones**: K²VAE, PPM, TMDM, Diffusion-TS, TimeDiff, NsDiff and DeepSTUQ. `list` prints exact task identities. Repeat `train` and `evaluate` for the required identities; run all three configured training runs for each configuration before summarizing:

```sh
python scripts/reproduce.py summarize --workspace "$PWD/work"
```

After installing the external dependencies, algorithm-level baseline checks can be run with `python -m pip install -r requirements-test.txt` and `python -m pytest experiments/icde_reliability/test_external_baselines.py -q`.

Native scores and all corrections share the complete native test iterator. Evaluation refits every calibration method from locally generated historical forecasts. `--external` adds EMOS, learned PIT (Qian et al., Technometrics 2025) and the 1000-tree ensemble-postprocessing QRF. Global DC, global PIT and conditional PIT are always included. Diffusion-TS uses the same unclipped sampler for Native and all corrections under standard-score normalization.

`--controls` adds partition resolution, fitting budget, temporal template fitting, source filtering and five shuffled state-assignment controls. Four mean/shape arms and final-FSPC fitting/deployment benchmarks are always evaluated. Retained-cohort ablations and cost/parity checks are stored separately from complete-test scores. See [the reproduction guide](docs/REPRODUCIBILITY.md) for outputs, seeds, datasets and experiment mapping.

## Apply FSPC to existing ensemble forecasts

If forecasts are already available, no backbone training is needed. Prepare three NPZ coordinate files as specified in [the array interface](docs/ARRAY_INTERFACE.md), then run:

```sh
python scripts/fspc_arrays.py \
  --train training.npz --validation validation.npz --test test.npz \
  --horizon 192 --seed 1 --output outputs/my_run --controls
```

This CPU-capable entry point calls the same `fit_unified`, four-arm reconstruction and `CompiledFSPC` implementations as the native route. Add `--external` after installing PyTorch and `requirements-external.txt` to compare EMOS, learned PIT and QRF.

## Code map

| Component | Source |
| --- | --- |
| Final FSPC fit, purged selection and refit | `experiments/icde_reliability/fspc_unified.py` |
| Affine/state mean and selection objective | `location_scale.py`, `mean_improved_all.py`, `mean_anchor.py` in the same directory |
| Residual pools, source filtering and fallback | `portable_dc.py`, `granularity.py` |
| Full compiled FSPC and strict event queries | `final_fspc.py` |
| Native-data reviewer adapter | `reviewer_native.py`, `reviewer_pipeline.py` |
| EMOS / learned PIT / QRF | `external_baselines.py`, `qian_pit.py`, `ragged_qrf.py` |
| Common metrics and three-run aggregation | `reliability_metrics.py`, `aggregation.py` |
| Original measured campaign runners | `run_*.py`, `aggregate_*.py`, and protocol documents |

The reviewer adapters remove dependencies on previously completed result directories; they do not replace the underlying model, FSPC or metric implementations. Original campaign runners remain available for inspecting their historical orchestration and parity checks. Their recorded-result replay modes require locally generated campaigns and are not the quick-start entry point.

Fresh training can vary across hardware and numerical libraries. Published metric summaries are not injected into computed outputs. This code-only branch does not distribute the historical prediction arrays or author-supplied aggregate corrections needed to reconstruct every paper cell exactly. See [third-party attribution](NOTICE.md) for source revisions and licenses.
