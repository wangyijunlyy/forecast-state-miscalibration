# FSPC coordinate-array interface

Use NumPy NPZ files with numeric arrays only. `scripts/fspc_arrays.py` supports CPU fitting and evaluation without native model imports. It opens the test file after all requested models have been fitted and their parameters saved.

Every training, validation and test file contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `rows` | `[N,100]` | Native ensemble members in the backbone's evaluation units |
| `targets` | `[N]` | Matching observed targets |
| `horizon` | `[N]` | Original zero-based forecast-step index, in `[0,H)` |
| `channel` | `[N]` | Original zero-based channel index |

`N` can differ between splits. Retain actual horizon/channel identities when sampling coordinates. The validation file additionally contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `window` | `[N]` | Original chronological validation forecast-origin index |
| `n_windows` | scalar | Total number of origins represented by the validation reservoir, including origins not retained by sampling |
| `origin_stride` | scalar | Actual chronological distance between adjacent native forecast origins, in target steps |
| `thresholds` | `[C,3]` | Historical per-channel target quantiles at 0.8, 0.9 and 0.95 |

Use all the validation historical targets for the target-threshold reservoir (capacity 4096 horizon-coordinate rows, preserving channels), as implemented by `TargetReservoir`. Thresholds and state boundaries must be fixed before looking at test targets. Native reproduction creates these arrays and provenance automatically.

```sh
python scripts/fspc_arrays.py \
  --train training.npz --validation validation.npz --test retained_test.npz \
  --horizon 192 --seed 1 --output outputs/task1 --controls
```

The default scores are Native, MeanOnly, ShapeOnly and full FSPC. `--controls` adds resolution/budget/temporal/source and state-assignment controls. `--external` adds EMOS, learned PIT and QRF using the measured external dependency versions. For native Global DC/global PIT/conditional PIT comparisons, use `scripts/reproduce.py evaluate`.

The returned `retained_results.json` describes exactly the supplied test coordinates; it is not a claim of full-test coverage. Complete fitting time excludes Native generation and file I/O. Post-fit compilation reuses the already fitted quantile grid. Query timings distinguish computing Native summaries from reusing those summaries. FSPC's strict event convention is `prediction > threshold`, including thresholds that equal a prediction value.

The deployment state is `compiled.npz`; load it with `CompiledFSPC.load(path)` and call `transform_rows(rows,horizon)` or `probabilities_rows(rows,horizon,thresholds_for_each_coordinate)`. The original FSPC reference parameters are saved separately in `parameters.json`.

For timing under the manuscript CPU-thread setting on Linux, set `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2` before starting Python and prefix the array command with `taskset -c 0,1`. Four concurrent measured workers used disjoint core pairs 0--1, 2--3, 4--5 and 6--7. Do not compare unpinned multi-threaded timings directly with those measurements.
