# FSDC 36-state default migration

Authorized on 2026-11-03: use six mean quantile groups and six log-scale quantile groups, pooling forecast horizons, as the manuscript default. Panel (b) uses the same 36-state basis: budgets 2k/10k, horizon groups 2/4/8 (72/144/288 states), and the unfiltered 36-state operator. Native is the denominator for average ratios.

## Evidence and scope

All 249 complete native checkpoints and corrected NsDiff h192 identities are retained. No neural network was retrained. New calibration and diagnostic evaluations consume complete native validation/test iterators with the original prediction seeds. Historical 16-state records remain unchanged. Primary scores are complete-test scores; retained-cohort sensitivity scores are reported separately.

Training- and validation-fitted 36-state calibration use the original 100-batch/window and 50k-coordinate limits. Training prediction and reservoir seeds are unchanged. Saved validation fitting reservoirs and event thresholds are reused because they do not depend on partition resolution. Source filtering, the 128-surviving-source fallback, 20k pool cap, 100 members, variance convention and 1e-6 scale floor are unchanged.

Every method uses the same training-fitted six-by-six diagnostic grid. Coverage and event summaries were recomputed on every complete test iterator. The campaign also recomputed training-fit 36/144 comparisons, budget and unfiltered controls, shuffled assignments and exact compilation benchmarks. Unchanged complete-test scores match the original campaign. Regenerated training reservoirs reproduce the legacy calibration boundaries and local/global pools exactly.

The 36-state default was adopted following the existing descriptive sensitivity study. It is not described as chosen before those test results or independently tuned on validation data. The study does not establish a universally optimal resolution.

## Completed execution ledger

- Server campaign: `icde_fsdc36_20261103`, separate from the original campaign. All 249 primary, assignment and compiled-deployment tasks are complete; all 249 cohort records contain 34 fits each (8,466 total). Every required suite audit reports no missing tasks or errors.
- New raw identifiers are DC36, TrainDC36, TrainDC144 and UnfilteredDC36. Formatters use temporary layout aliases only; they never relabel legacy scores. Supplementary CSV retains DC36 keys.
- Two eight-worker GPU queues (original PID 1029168, auxiliary PID 1037615) drained naturally. A per-task lock and completion recheck prevent duplicate work. All 249 task logs contain exactly one completion marker. A patched final controls-driver invocation exited zero after all primary tasks completed, resolving the earlier terminal-drain race.
- The server has 64 CPU threads and 251 GiB RAM. Measured GPU allocations stayed below 5.4 GiB/task, with at most two workers/GPU. Runtime measurements describe this shared server workload. Fitting time starts from already generated fitting ensembles; it excludes their native forecast generation.
- Twenty-two numerical tests pass in Python 3.10, NumPy 1.26.4 and PyTorch 2.4.1+cu121. Twenty NumPy tests also pass locally. They check legacy parity, six-by-six fitting, 144-state horizon reconstruction, state-35 populations, fallback, metric arithmetic, sample SD and compiled queries. Native NLL checks ran on the server.
- The portable CLI defaults to six summary groups. Its numeric export exactly matches actual PPM and DeepSTUQ 36-state calibrators. The Python fitter preserves its historical four-group default; request `summary_bins=6` explicitly.
- Primary manuscript comparisons preserve five methods: Native, FSDC, Global DC, Global PIT and Conditional PIT. Early optional Gaussian outputs remain recorded, but Gaussian controls are not a complete-suite claim or a completion gate.
- Canonical main/supplement tables, dependent prose, diagnostics and source figures now use the audited 36-state results. Independent audits verify 493 main and 830 supplementary mean/SD cells, 2,016 supplementary coverage cells and 14 reconstructed source cases. All 22 supplementary figures use the current registry and provenance.
- Current main body: 12 pages; supplement: 27 pages. The updated code/record supplement is local only. Existing public v1.2.0 remains the historical release.

## Aggregation and record replay

Within a run, average configurations per backbone before dividing by the corresponding Native backbone mean. Average those ratios equally across seven backbones, then report the mean and sample SD across three runs. Do not average absolute scores across backbones with different units.

`rebuild_analysis.py --fsdc36-records results/fsdc36` invokes complete-suite auditing and table rendering. See `README.md` for commands from the workspace root, including figure preparation, source auditing and PDF builds.
