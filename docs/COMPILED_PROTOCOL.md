# Exact fixed-grid deployment and threshold queries

This is an implementation experiment for the completed237-task primary suite. It changes neither fitted residual pools nor any predictive-quality result. Compile each validation-fitted DC16 pool into the same100 midpoint quantiles used in the actual reconstruction. Share fallback templates through a16-state-to-template lookup. Deploy only quantile tables, state boundaries and horizon metadata.

For a threshold query, binary-search the ordered reconstructed values `mu + sigma*q[j]`, returning the fraction strictly above the threshold. Comparing reconstructed values directly preserves floating-point ties that can change when a threshold is instead standardized. Reusing stored native ensemble means and ddof1/floored scales allows repeated threshold queries without retaining or decoding a100-member corrected ensemble.

Evaluate all237 completed identities on their saved uniform test-coordinate reservoirs (at most50000 coordinates, retained during each complete native test traversal). Use each coordinate's original horizon/channel IDs and the actual three historical per-channel thresholds. The runtime suite is separate from the complete-test predictive-quality suite.

Before timing, compare all100 reconstructed values and all three event probabilities with the original saved calibrator. Recompute event Brier scores and compare with the existing50k-budget reservoir control. No coordinate is removed. Warm each implementation once through the parity calculations. Time three repetitions in a randomized order per task, with deterministic schedule seed20261102 plus training seed; retain individual timings and use their median.

Operations: original full decoding; compiled full decoding; original decoding plus three event queries; direct queries including native summary computation; direct queries reusing the same precomputed summaries. Native neural generation, file I/O and metric calculation are outside these timing intervals. Record compilation time separately, original single-calibrator pickle bytes, compiled uncompressed NPZ bytes, workload size and exact probability checks. CPU timings describe the actual parallel server campaign; report the hardware/software environment.

Only after all237 task records pass numerical parity can backbone timing/storage summaries be reported. Average configurations within each training seed before computing three-seed mean/sample SD. Manuscript standard deviations report measured sample SD rounded to three decimals; raw statistics retain full precision.
