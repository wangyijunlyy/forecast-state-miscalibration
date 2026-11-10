# Source attribution

Native implementations, datasets and third-party libraries retain their original licenses and terms. The repository preserves available license files in `native_payload`; public restoration uses the source identity and exact file manifest in `upstream_source_reconstruction/source_reconstruction.json`.

| Component | Source identity |
| --- | --- |
| K²VAE training/evaluation integration | ProbTS/K²VAE-derived native modules; training snapshot based on `wangyijunlyy/ProbTSF-NLL`, revision `fc8da55ea4df5ddd83c0fa28fe91704ca1c637d0` |
| PPM | `https://github.com/ljl8336/PPM`, revision `6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9`; native loader/evaluation integration included |
| NsDiff / TMDM / Diffusion-TS / TimeDiff integration | `https://github.com/wwy155/NsDiff`, public base revision `dd7b97ae0c47629986f72ba01effbbdc1d44be2b` plus the supplied measured patches |
| DeepSTUQ | `https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch`, revision `4f2281531d07b48ba81aa1c519a90e65a71993e2` |
| QRF adapter | Full bootstrap-leaf representation in `ragged_qrf.py`, with weighted-quantile interpolation corresponding to BSD-3-Clause `quantile-forest` 1.4.2 |

Paper baselines are Gaussian EMOS/NGR; Qian et al., *Regression Recalibration by Learning PIT Map Values*, Technometrics (2025); and Taillardat et al., *Calibrated Ensemble Forecasts Using Quantile Regression Forests and Ensemble Model Output Statistics*, Monthly Weather Review (2016). Their implementation choices are documented in `docs/EXTERNAL_BASELINES_PROTOCOL.md` and the source metadata. Distribution recalibration code is retained as an optional historical experiment; it is not included by the reviewer's `--external` option for the current manuscript comparison.

Local wrappers preserve native architecture/training settings and isolate output directories. Numerical changes for memory-bounded diffusion guidance, attention overflow fallback and corrected unclipped Diffusion-TS inference are explicit in their source and protocol documents. The FSPC reviewer adapters add fresh-data orchestration and do not read or substitute archived scores.

The optional Hugging Face `datasets` import in the two GIFT loader modules is deferred until that loader is used. The benchmark uses the native LTSF/traffic loaders; this import-only adjustment does not change their preprocessing or forecast computation.
