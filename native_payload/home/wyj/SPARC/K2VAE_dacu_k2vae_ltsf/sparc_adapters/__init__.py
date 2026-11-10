"""Model-independent SPARC adapters for external probabilistic forecasters."""

from .metrics import evaluate_samples_like_k2vae, write_horizons_results_csv
from .nsdiff import NsDiffSPARCAdapter
from .ppm import PPMSPARCAdapter
from .calibration import SampleSPARCCalibrator
from .samples import (
    normalize_sample_layout,
    sample_preserving_rescale,
    summarize_canonical_samples,
)

__all__ = [
    "evaluate_samples_like_k2vae",
    "write_horizons_results_csv",
    "NsDiffSPARCAdapter",
    "PPMSPARCAdapter",
    "SampleSPARCCalibrator",
    "normalize_sample_layout",
    "sample_preserving_rescale",
    "summarize_canonical_samples",
]
