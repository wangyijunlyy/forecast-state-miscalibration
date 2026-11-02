# Forecast-state reliability: reproduction artifact

This artifact contains the code, configurations, 237 native forecasting checkpoints, 12 fully trained DeepSTUQ checkpoints, required conditional predictors and shared initializations, native dataset files, training-fitted residual pools, reference records, and scripts for rebuilding the paper's numerical tables and two figures. The manuscript PDF and author information are distributed separately. Public hosting is a distinct publication step; the package does not claim an available URL before hosting is verified.

## Extract and rebuild the reported analysis

Extract the code/records archive, `base-assets.tar.gz`, and `traffic-assets.tar.gz` into the same new directory. Asset filenames and sizes are listed in the accompanying archive manifests. Extract the package at its final location before relocation. Table and figure generation needs Python, NumPy, SciPy, and Matplotlib; it does not require a GPU or model inference:

```sh
python experiments/icde_reliability/rebuild_analysis.py --workspace "$PWD"
```

This independently audits all 237 primary, 237 Gaussian, 237 compiled, 237 state-assignment, and 12 traffic records, then rebuilds `ICDE/tables/` and `ICDE/figures/`. `analysis_summaries/` contains the recalculated numerical summaries. Aggregation averages configurations within each training seed before computing the mean and sample SD across seeds 1, 2, and 3. All manuscript tables report the measured sample SD rounded to three decimals; raw statistics retain full precision. Integer counts, event totals, and paired population checks support the reported metrics. Rebuilding tables from records and recomputing forecasts are separate operations.

## Restore the native model sources and environment

The public package supplies verified public source revisions and the measured source changes for 505 upstream files. Fetch those sources before relocation:

```sh
python restore_upstream.py --bundle "$PWD"
python relocate_eab.py --bundle "$PWD"
python3.10 -m venv .venv
. .venv/bin/activate
python setup_environment.py --requirements requirements-runtime.txt
. ./activate.sh
python check_relocated.py --bundle "$PWD"
```

The bootstrap requires network access to the pinned public repositories and the standard `patch` utility. Attribution and available upstream licenses are retained; upstream repositories are not relicensed. `payload/` preserves the loader layouts. Relocation changes only copied source/configuration/metadata text and never rewrites model binaries or data. Dataset copies are separate from live experiment inputs because some native loaders re-extract files. A relocated tree refuses relocation to a different root; use a pristine extraction when moving it.

The measured native environment is Linux, Python 3.10.19, PyTorch 2.4.1+cu121, NumPy 1.26.4, SciPy 1.15.3, and RTX4090 GPUs. The installer first obtains the pinned CUDA wheels, then installs the full resolved requirements and runs `pip check`. The compatible 174-distribution lock uses packaging 24.2 and fsspec 2024.12.0 to satisfy Lightning constraints while retaining the original numerical/model-library versions. Original and resolved dependency records are kept separately in provenance. The seven native interfaces pass imports in the clean environment.

## Recompute native forecasts and paired scores

Select a task identity from the manifest:

```sh
python replay_eab.py --bundle "$PWD" --stage quality \
  --task K2VAE_ettm1_h192_seed1 --gpu 0
```

The command loads the supplied native checkpoint, traverses the complete native validation iterator, fits validation corrections, and traverses the complete native test iterator. It recreates six paired quality outputs, validation diagnostics, event thresholds, and retained-coordinate controls. Active replay outputs start empty and are separate from `reference_records/`; the command uses no copied quality-result cache. Repeat for every manifest identity to recompute the primary suite.

The earlier training-fitted residual state is supplied as an input. Independently regenerate and compare its DC16/DC64 arrays with:

```sh
python replay_eab.py --bundle "$PWD" --stage traincal \
  --task K2VAE_ettm1_h192_seed1 --gpu 0
```

Original backbone training writes a fresh run rather than skipping a supplied checkpoint:

```sh
python replay_eab.py --bundle "$PWD" --stage train \
  --task K2VAE_ettm1_h192_seed1 --gpu 0
```

For the Gaussian controls, run after the matching primary quality replay:

```sh
export ICDE_RUN_ROOT="$PWD/payload/home/wyj/icde_reliability_20261102"
export CUDA_VISIBLE_DEVICES=0
python "$ICDE_RUN_ROOT/code/run_gaussian.py" \
  --task "$ICDE_RUN_ROOT/replay_tasks/K2VAE_ettm1_h192_seed1.json"
```

This refits both Gaussian models on the regenerated validation reservoir and repeats complete native test evaluation. After all primary tasks, run the retained-coordinate state and execution controls:

```sh
python "$ICDE_RUN_ROOT/code/benchmark_compiled.py" --root "$ICDE_RUN_ROOT"
python "$ICDE_RUN_ROOT/code/run_state_controls.py" --workers 4
```

Their timed cohorts are separate from complete-test quality evaluation. `PROTOCOL.md`, `GAUSSIAN_PROTOCOL.md`, `STATE_CONTROLS_PROTOCOL.md`, and `COMPILED_PROTOCOL.md` specify the fitting budgets, random states, metric conventions, and comparisons. Independent aggregate scripts accept explicit regenerated run directories; their command-line options are shown with `--help`. Pass the regenerated records to `rebuild_analysis.py --records` using the same layout as `reference_records/` to regenerate the manuscript analysis from new outputs.

## Recompute the original ICDE traffic method

```sh
python replay_eab.py --bundle "$PWD" --stage deepstuq \
  --dataset PEMS04 --seed 1 --gpu 0
```

The default loads the supplied fully trained native model and temperature, refits empirical corrections, and evaluates the complete native traffic test iterator. Add `--from-scratch` with a fresh `--output-root` to run all native pretraining (at most 100 epochs, patience 15), all 20 AWA epochs and ten averaged snapshots, ten dropout inference passes, and native temperature calibration. The wrapper bypasses only the old campaign queue-wait hook. It preserves the author architecture, substantive objectives, and all training/calibration stages.

Training-only normalization and the 100-midpoint representation of the native Gaussian marginal are specified in `DEEPSTUQ_PROTOCOL.md`. The 100 quantiles do not denote 100 dropout passes or joint trajectories. Full native Gaussian NLL, 95% coverage, and interval-width scores are retained. The traffic suite uses flow units on its own matched tasks; absolute scores are not ranked against the long-horizon suite.

## Executed replay evidence

Public-source restoration was executed and all 505 restored files matched the observed sources byte for byte. In a fresh Python environment and a fresh relocated tree, one complete validation/test task per long-horizon backbone reproduces 216 primary-method metric comparisons exactly. Refitting Gaussian2P/Gaussian4P and repeating those complete native tests reproduces 72 further comparisons exactly. A complete native PEMS04 traffic replay reproduces 33 comparisons, including five paired methods and native Gaussian scores, exactly. These are 321 distinct metric comparisons, with maximum absolute difference zero. The training-fitted K2VAE DC16/DC64 residual arrays have also been regenerated exactly. Detailed audits are retained in provenance.

The legacy SPARC namespaces supply checkpoint and data interfaces. The experiment obtains forecasts from the raw backbone samplers and applies the evaluated calibration operators explicitly. Unused CAISO, Nordpool, Kaggle, and Solar-NIPS inputs are excluded from the package. Figure extraction retains measured source arrays; its example-selection rule uses validation templates and forecast outputs rather than test-target selection.
