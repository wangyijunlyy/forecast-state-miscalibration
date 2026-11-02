# ICDE 2023 DeepSTUQ baseline

This document records the original full native training replication. The current paper treats DeepSTUQ as its seventh backbone and applies the full common experiment matrix on all four original datasets; see `DEEPSTUQ_BACKBONE_PROTOCOL.md`. The original five-method records remain as the native replay reference, rather than a separate final comparison table.

User-requested additional baseline, selected after locating an actual probabilistic forecasting method from a core data engineering conference.

Paper: Weizhu Qian, Dalin Zhang, Yan Zhao, Kai Zheng, James J. Q. Yu, *Uncertainty Quantification for Traffic Forecasting: A Unified Approach*, ICDE 2023, pp.992--1004, DOI 10.1109/ICDE55515.2023.00081.
Author code: https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch ; inspected revision 4f2281531d07b48ba81aa1c519a90e65a71993e2.
Primary paper: https://file.jqyu.me/pub/uncertainty-quantification-for-traffic-forecasting-a-unified-approach.pdf .

## Full method identity

Use the original combined AGCRN_UQ architecture, two 64-unit recurrent layers, original graph convolutions and learned node embeddings. Preserve all original per-dataset model configurations. PEMS03/04/07/08, one-hour context and horizon (12 five-minute positions), chronological 60/20/20 split. Seeds 1,2,3 are independent training replications.

Pretraining: Adam lr0.003, weight decay1e-6, original per-dataset batches (64 for PEMS03/04/08 and32 for PEMS07), at most100 epochs, validation selection and original patience15. Combined Gaussian NLL/MAE objective with weights0.1/0.9. AWA: all20 retraining epochs, ten averaged snapshots, Adam lr high0.003/low0.00003, original per-batch alternating schedule. Inference: ten dropout forward passes. Preserve decoder dropout0.2 and encoder probabilities0.1 for03/04/07 and0.05 for08. Native temperature calibration uses validation data, original log-temperature parameter and LBFGS lr0.02/max_iter500. Predictive Gaussian mean and variance follow original `combined_test`: mean of MC means, sample variance of MC means plus exp(mean log variance)/exp(log temperature).

Comparisons: the full DeepSTUQ native distribution (including native temperature calibration), the same backbone distribution before temperature calibration as a labelled component ablation, and state-conditioned source-filtered distribution calibration fitted on historical validation forecasts. Also compare global empirical residual calibration and PIT recalibration on the identical predictive marginals. This native traffic experiment is a distinct matched suite; do not directly rank its absolute scores against the six-backbone long-horizon benchmark on different datasets.

The native Gaussian distribution is exposed through100 equal-probability midpoint quantiles for common empirical CRPS/QICE/state/event metrics. Also retain original Gaussian NLL,95% interval coverage and width. Native NLL follows the author's PyTorch `GaussianNLLLoss` defaults: omit the constant0.5log(2pi) and clamp variance at1e-6 for that loss only. Report this deterministic marginal representation explicitly rather than claiming100 extra dropout passes or joint trajectory samples.

## Source integration issues identified before running

The author `main.py` contains an undefined `train_cali_mcx` import/call mismatch, comments out AWA, and ignores command-line arguments. The AWA function references an undefined global `model`; the wrapper must bind it to the actual trainer model and invoke all20 epochs. Test printing has undefined point-metric variables; compute those explicitly. The source normalizes before temporal splitting; use train-only statistics in this experiment and document the preprocessing difference. Preserve architecture and substantive objectives/stages; do not present a missing-AWA shortcut as DeepSTUQ.

Reuse the untouched author source through a wrapper and record the above interface repairs. Missing author code license means the final artifact should provide a fetch command and revision rather than assume permission to redistribute the third-party source.

## Required evidence

Twelve completed checkpoint records; configuration and actual training histories; all20 AWA epochs and averaged snapshot count; fitted temperature; full-test counts; paired raw/calibrated marginal scores; native-format metrics; exact source revision and actual dataset shapes; independent arithmetic and source-code checks.

## Execution scheduling

Full baseline training can begin on GPU cards drained by the primary suite after its pending queue reaches zero. Cards occupied by a live primary worker remain reserved; no card is shared by two workers. Native baseline calibration and test inference begin only after all237 primary results and processes finish. This scheduling update changes no data, model, objective, epoch budget, calibration setting or scoring rule. CPU timing measurements in the main suite describe the concurrent server campaign.
