import numpy as np

from sparc_ppm.posthoc import PPMMetricAccumulator
from sparc_ppm.ppm_metrics import evaluate_ppm_samples


def test_streaming_metrics_match_list_evaluator():
    rng = np.random.default_rng(17)
    samples = [rng.normal(size=(2, 100, 4, 3)), rng.normal(size=(3, 100, 4, 3))]
    targets = [rng.normal(size=(2, 4, 3)), rng.normal(size=(3, 4, 3))]
    expected = evaluate_ppm_samples(samples, targets)
    accumulator = PPMMetricAccumulator()
    for prediction, target in zip(samples, targets):
        accumulator.update(prediction, target)

    actual = accumulator.result()

    for metric in ("CRPS", "QICE", "MSE", "MAE"):
        np.testing.assert_allclose(actual[metric], expected[metric], rtol=1e-12, atol=1e-12)
