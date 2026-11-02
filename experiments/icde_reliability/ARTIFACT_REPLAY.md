# Reproduce from code, data and recorded results

The public release contains code, configurations, reference results and datasets. Neural model weights are excluded at the author's request. Training creates model checkpoints locally.

## Rebuild the reported tables and figures

Extract `code-and-records.tar.gz` into a new directory. With Python3.10–3.12 and NumPy1.26.4,SciPy1.15.3,Matplotlib3.10.3 installed:

```sh
python experiments/icde_reliability/rebuild_analysis.py --workspace "$PWD"
python experiments/icde_reliability/audit_table_sd.py --results analysis_summaries --paper ICDE
```

This audits the saved experiment records,rebuilds the numerical tables and two figures,and independently checks all525 mean/SD cells against raw three-seed scores. It does not require datasets,a GPU or trained model parameters. Raw statistics retain full precision;manuscript tables use measured sample standard deviations to three decimal places.

## Restore data and native sources

Concatenate the data parts in their three-digit suffix order and extract into that same directory:

```sh
cat data-only.tar.gz.part-[0-9][0-9][0-9] > data-only.tar.gz
tar -xzf data-only.tar.gz
python restore_upstream.py --bundle "$PWD"
python relocate_eab.py --bundle "$PWD"
python3.10 -m venv .venv
. .venv/bin/activate
python setup_environment.py --requirements requirements-runtime.txt
. ./activate.sh
```

The measured native environment is Linux,Python3.10.19,PyTorch2.4.1+cu121,and RTX4090 GPUs. Public upstream revisions plus measured patches reconstruct the model sources. Available upstream licenses and source attribution are retained.

## Train and evaluate a native task

From the fresh relocated extraction:

```sh
python from_scratch_eab.py --bundle "$PWD" \
  --task K2VAE_ettm1_h192_seed1 --gpu 0 --check-only
python from_scratch_eab.py --bundle "$PWD" \
  --task K2VAE_ettm1_h192_seed1 --gpu 0
```

Use the manifest identities for the other backbone,dataset,horizon and seed combinations. The command invokes the original native training entry point with the recorded epoch budgets and validation selection,collects native training forecasts to refit the source-filtered DC16/DC64 residual pools,and evaluates all six primary methods on complete native validation/test iterators. Stages may also be run individually with `--stage train`, `--stage fit`, and `--stage quality`. Generated checkpoints and calibration states remain local. Native output directories must start without model checkpoints for a new training replication.

NsDiff on ETTh1 uses shared seed1 mean/variance initializers. Before diffusion training,the helper invokes the native F/G pretraining classes with their actual saved arguments (`provenance/shared_initializers/F.json` and `G.json`):F uses ten epochs and G uses twenty,with their original validation selection settings. The initializers are trained once locally and reused across diffusion seeds,as in the experiment protocol. The wrapper omits only the stray `%ln` notebook-magic line when loading the native G class. Model architectures and training objectives are unchanged.

For Gaussian controls,run the supplied Gaussian runner against the generated replay task JSON after primary quality evaluation. Compiled and state-assignment scripts consume the regenerated primary calibration/test reservoirs. Their native fitting budgets,random states and score definitions are specified in the accompanying protocols.

## Train and evaluate DeepSTUQ as the seventh backbone

```sh
python replay_eab.py --bundle "$PWD" --stage deepstuq \
  --dataset PEMS04 --seed 1 --gpu 0 --from-scratch
```

Repeat for PEMS03/04/07/08 and seeds1,2,3. This executes the full original combined architecture,native pretraining,all twenty AWA epochs and ten snapshots,ten dropout inference passes,and validation-fitted temperature calibration. The protocol records the training-only normalization and 100-midpoint Gaussian marginal representation used for common metrics.

After each native training run, execute all common calibration and control experiments against its locally generated checkpoint:

```sh
CUDA_VISIBLE_DEVICES=0 python "$ICDE_RUN_ROOT/code/deepstuq_backbone.py" \
  --dataset PEMS04 --seed 1 \
  --native-runs "$ICDE_RUN_ROOT/standalone_traffic/deepstuq/runs" \
  --output-root "$ICDE_RUN_ROOT/retrained_deepstuq_backbone"
```

Repeat both steps for all four datasets and three seeds. This second command fits training DC16/DC64,compares the same eight primary/Gaussian methods on the complete native test iterator,and runs the fitting-budget,source-validity,state-assignment and exact-compilation controls. See `DEEPSTUQ_BACKBONE_PROTOCOL.md`. No additional neural training is performed by this command when the native checkpoint is complete.


## Reference records and new training runs

The saved records document249 paired checkpoints:237 long-horizon tasks and twelve DeepSTUQ tasks. All seven backbones receive the same experiment families and appear in shared tables. Complete native replay with the original private checkpoints previously reproduced321 metric comparisons exactly;those checkpoints are not distributed in this release. Public reconstruction of the reported tables uses the saved records. New native forecast replications use locally trained checkpoints and should be compared with the recorded scores under the same seed-aware aggregation. Original supplied-checkpoint shortcuts require privately retained checkpoints. This public release uses the from-scratch route above.

## NsDiff192-step tasks

The current manifest replaces NsDiff ETTm1,ETTm2 and ExchangeRate24-step tasks with192-step tasks. Use identities such as `NsDiff_ETTm1_h192_seed1` with `from_scratch_eab.py`. Its corrected-task branch trains the original native networks and invokes `run_nsdiff_h192.py` for training DC16/DC64,all eight primary/Gaussian comparisons and all mechanism/compilation controls. Outputs are under `$ICDE_RUN_ROOT/retrained_nsdiff_h192`. Other native hyperparameters remain fixed; see `NSDIFF_H192_PROTOCOL.md`.

The nine replacements are selected by `experiments/icde_reliability/current_manifest.json`. Historical24-step records remain archived but do not contribute to current249-task summaries.

`train_nsdiff_h192.py` retains the native training architecture and optimizer,uses an equivalent attention float64 overflow fallback,and treats nonfinite validation candidates as non-improvement under the original patience10 rule. `NSDIFF_H192_PROTOCOL.md` documents the observed case. The granularity records contain8217 partition/budget fits on the corrected249 tasks. Table SD is shown as a right subscript to the mean,with three decimals and bold best unrounded means.
