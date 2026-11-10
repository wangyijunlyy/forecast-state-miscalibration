import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_utils import _early_stopping_enabled, _make_early_stopping_callback


class DummyEarlyStopping:
    def __init__(self, monitor, patience, mode, min_delta, verbose):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose


def test_early_stopping_disabled_by_default():
    assert _early_stopping_enabled(None) is False


def test_early_stopping_callback_uses_monitor_and_patience():
    callback = _make_early_stopping_callback(
        {
            "enabled": True,
            "monitor": "val_CRPS",
            "patience": 3,
            "mode": "min",
            "min_delta": 0.001,
        },
        fallback_monitor="val_weighted_ND",
        callback_cls=DummyEarlyStopping,
    )

    assert callback.monitor == "val_CRPS"
    assert callback.patience == 3
    assert callback.mode == "min"
    assert callback.min_delta == 0.001
