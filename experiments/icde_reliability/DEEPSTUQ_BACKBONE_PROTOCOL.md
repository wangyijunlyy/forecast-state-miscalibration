# Seventh-backbone experiment coverage

Dataset scope follows Section V-A of *Uncertainty Quantification for Traffic Forecasting: A Unified Approach* (ICDE2023): PEMS03/04/07/08, twelve historical five-minute steps and twelve forecast steps, chronological60/20/20 split. Three independent training seeds are retained for each dataset. The complete author architecture, native pretraining and all20 AWA epochs/ten snapshots, ten dropout passes and validation temperature fit are unchanged. This study uses training-only normalization, as documented in the original replication protocol.

DeepSTUQ now receives the same experiment families as the six long-horizon backbones:

1. Training-fitted DC16/DC64 from at most100 native training batches and a uniform50000-coordinate reservoir, using prediction seed `training_seed+10000` and reservoir seed `training_seed`.
2. Full native validation traversal, with Native/TrainDC16 diagnostics on the training-fitted mean/scale partition. The first100 native validation batches supply the common validation fitting reservoir. PIT, residual-source rules, boundaries, minimum counts, numerical floors and sample conventions use the existing implementation without changes.
3. Complete native test evaluation of Native, TrainDC16, ValDC16, GlobalDC, GlobalPIT, ConditionalPIT, GaussianVariance2P and GaussianAffine4P on identical forecasts. Gaussian controls receive exactly the same validation coordinates. Training-fitted DC64 is also scored on the complete test iterator for the configuration-level partition comparison. Prediction seed is `training_seed+30000`. Native scalar metrics must match the earlier full DeepSTUQ replication.
4. CRPS, QICE, mean/median point scores, common-state coverage, and three per-channel historical exceedance probabilities/Brier scores, using the same metric code.
5. Nested2k/10k/50k calibration budgets, validation-fitted DC64 and the unfiltered-source ablation, evaluated on a shared uniform test-coordinate reservoir retained during complete traversal.
6. Five residual-to-state assignment shuffles, preserving the source mask, boundaries, state counts, fallback eligibility and global pool.
7. Exact compiled reconstruction and three exceedance queries, including three randomized paired runtime repetitions, serialized/numeric state size and compilation time. All100 values and all event probabilities must match the original correction.
8. Native generation, fitting, correction and metric time, CPU/GPU peak memory, and equal-dataset-within-seed aggregation followed by sample SD across seeds1/2/3. Table values use three decimals.

The native predictive Gaussian is represented by100 deterministic midpoint quantiles; these are not100 dropout draws. DeepSTUQ scores retain traffic-flow units, while the other backbones retain their model-normalized units. Each backbone summarizes its own matched tasks. No absolute cross-backbone ranking pools these two scales.

Primary, Gaussian, state-assignment and compiled-deployment auditors consume the twelve-task extension manifest. An additional auditor checks the native full-training prerequisites, train/validation fitting identities and exact original Native-score parity. Public reproduction starts from native training code and original data; neural model weights remain excluded.
