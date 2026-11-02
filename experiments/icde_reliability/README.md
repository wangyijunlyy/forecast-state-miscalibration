# Forecast-state calibration experiments

The complete seven-backbone experiment is specified in `PROTOCOL.md` and `DEEPSTUQ_BACKBONE_PROTOCOL.md`; full native DeepSTUQ training is specified in `DEEPSTUQ_PROTOCOL.md`. The primary runner reuses the actual trained checkpoint interfaces and consumes their full native validation/test iterators. It is distinct from the portable array evaluator below.

## Fit and apply the complete empirical correction

`portable_dc.py` is the NumPy implementation of source-filtered DC16 and DC64. Fitting takes the retained historical coordinate reservoir: `rows` `[N,100]`, `targets` `[N]`, and original `horizon` indices `[N]` in an NPZ file. The experiment constructs that reservoir from at most100 native fitting batches/windows and retains up to50000 coordinates. The portable fitter receives those coordinates directly.

```sh
python portable_dc.py fit --coordinates calibration.npz --horizon-length 192 \
  --horizon-bins 1 --output dc16.npz
python portable_dc.py apply --calibrator dc16.npz \
  --samples native_samples.npz --output corrected_samples.npz
```

`native_samples.npz` contains `samples` `[B,100,H,C]`. Use `--horizon-bins 4` for DC64. The exported calibrator stores boundaries and all selected residual pools as numeric arrays; loading needs no pickle, model class or GPU. `EmpiricalCalibrator.transform_rows(rows, horizon_ids)` also supports arbitrary retained coordinate order while preserving actual forecast positions.

The deployment code was compared with12 actual server calibrators across all six backbones, including collapsed-source cases in PPM and Diffusion-TS. Boundaries and local/global residual pools match exactly. Original and portable reconstruction agree within float64 rounding (maximum absolute difference1.78e-15 in the checked forecasts). Raw evidence is `results/portable_dc_parity.json`; `audit_portable_dc.py` reruns that comparison against completed server records.

## Evaluate outputs from another forecasting pipeline

`evaluate_stream.py` needs Python and NumPy. It consumes one or more NPZ batch files. Every file contains:

- `targets`: numeric array `[batch, horizon, channels]`.
- `Native`: native ensemble `[batch, 100, horizon, channels]`.
- One array of that identical shape per additional method, named by the method, e.g. `DC16` or `GlobalPIT`.

The method set and horizon/channel dimensions must remain the same across files. All methods use the state of `Native`, the same targets, and the same historical event thresholds. Scores cover every supplied batch; the caller is responsible for supplying the complete desired evaluation split.

Supply `history.npz` with historical `targets` retaining the final channel dimension, and a separately fitted `states.json` containing the three interior mean and log-scale cuts:

```json
{"mean_cuts": [-0.5, 0.0, 0.5], "log_scale_cuts": [-2.0, -1.0, 0.0]}
```

The numbers above illustrate the file structure. Use the actual boundaries fitted on historical forecasts, not these illustrative values. Thresholds are historical per-channel target quantiles at 0.8, 0.9, 0.95, estimated using the same uniform 4096-row reservoir procedure. Default reservoir seed is 50001.

```sh
python evaluate_stream.py --history history.npz --states states.json \
  --output scores.json batch000.npz batch001.npz
```

The output includes empirical CRPS, QICE in proportion and percentage-point units, mean/median point errors, all 16 state populations and coverage counts, eligible-state diagnostics, three event Brier scores, and ten-bin event reliability summaries. No model-specific library or GPU is needed to evaluate exported arrays. The interface evaluates predictive marginals.

## Audit and aggregate actual experiment records

From the repository root:

```sh
python experiments/icde_reliability/aggregation.py \
  --runs experiments/icde_reliability/results/runs \
  --manifest experiments/three_seed_reproduction/manifest.json \
  --output experiments/icde_reliability/results/summary
```

The independent auditor checks manifest identities, six paired methods, complete traversal flags, common state/event populations, coverage arithmetic, Brier totals and bounds, and Native parity against historical scores. It records missing tasks explicitly. Configuration summaries require seeds 1, 2, and 3. Backbone statistics average configurations within each seed, then compute mean and sample SD across the three seed averages. A state diagnostic may be undefined if no state has128 test coordinates. That diagnostic uses the common subset of configurations with defined values in all three seeds, with its denominator recorded per metric; ordinary scores retain all configurations. Since all methods share the same state counts, this subset is paired across methods. `partial_backbone_statistics.csv` remains a progress snapshot until `audit.json` reports all manifest tasks complete. Raw CSV retains actual standard deviations; manuscript cells display measured sample SD rounded to three decimals.

```sh
python -m unittest discover -s experiments/icde_reliability -p 'test_*.py' -v
```

## Server execution

`launch.py` schedules the primary experiment with one worker per GPU. Once its pending queue empties, `launch_deepstuq.py` can start full baseline training on drained cards, reserving every card still occupied by a live primary worker. The baseline's calibration and test inference wait for all237 primary tasks and processes to finish. Each task records training identity, fitting split, iterator counts, generation/postprocessing/metric timing and memory. Native generation uses synchronized CUDA timing; CPU postprocessing and metric computation are recorded separately. Resource timings describe the parallel server campaign rather than isolated single-process CPU measurements.

The author DeepSTUQ repository is fetched at the revision specified in `DEEPSTUQ_PROTOCOL.md`. Third-party source redistribution depends on its license; the reproduction package should fetch the original revision rather than silently relicense that source.

`collect_provenance.py` preserves the actual source/configuration dependencies and package versions used on the server. Its private snapshot includes local interface corrections and source folders that have no Git metadata. `provenance/source_provenance.json` identifies revisions and license files, and `provenance/environment.json` records the installed environment. The private source snapshot is separate from a public release; code without a redistribution license is obtained from its original repository by the eventual reproduction instructions.

## Seventh-backbone integration

DeepSTUQ uses all four original PeMS datasets and three training seeds. `deepstuq_backbone.py` reuses each full native checkpoint and evaluates every common calibration/control family. `rebuild_analysis.py` combines its twelve tasks with the237 long-horizon tasks, independently audits all249 records per suite, and renders shared seven-backbone tables. `audit_deepstuq_backbone.py` additionally checks complete native training, original-output parity, full traversal, fitting identities and source counts. Traffic-flow scores retain native units; long-horizon scores retain their native model-normalized units.

## Corrected NsDiff horizons

`current_manifest.json` selects the current long-horizon tasks. NsDiff ETTm1,ETTm2 and ExchangeRate use192 forecast steps and replace the nine historical24-step training-seed identities. `NSDIFF_H192_PROTOCOL.md` records the unchanged native hyperparameters and complete recalculation. `run_nsdiff_h192.py` evaluates the same correction/control families from newly trained checkpoints; `audit_nsdiff_h192.py` checks actual output/fitting horizons and paired native metric accumulation. `rebuild_analysis.py` selects these nine replacements before combining all seven backbones,keeping249 tasks and83 configurations.
