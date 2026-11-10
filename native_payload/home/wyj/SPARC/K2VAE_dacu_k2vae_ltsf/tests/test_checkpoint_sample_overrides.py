import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SAVE_UTILS = ROOT / "probts" / "utils" / "save_utils.py"
RUN_PY = ROOT / "run.py"


def load_save_utils_module():
    forecaster_module = types.ModuleType("probts.model.forecaster")

    class Forecaster:
        pass

    forecaster_module.Forecaster = Forecaster
    sys.modules.setdefault("probts", types.ModuleType("probts"))
    sys.modules.setdefault("probts.model", types.ModuleType("probts.model"))
    sys.modules["probts.model.forecaster"] = forecaster_module

    spec = importlib.util.spec_from_file_location("save_utils_for_test", SAVE_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyForecaster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.no_training = False


class RecordingModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_state_dict(self, state_dict):
        self.state_dict_loaded = state_dict


def test_load_checkpoint_allows_sample_count_overrides(tmp_path):
    save_utils = load_save_utils_module()
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint = {
        "hyper_parameters": {
            "forecaster": {
                "class_path": f"{__name__}.DummyForecaster",
                "init_args": {},
            },
            "num_samples": 100,
            "val_num_samples": 100,
            "learning_rate": 0.001,
            "quantiles_num": 20,
            "load_from_ckpt": None,
        },
        "state_dict": {"weight": torch.tensor(1.0)},
    }
    torch.save(checkpoint, checkpoint_path)

    model = save_utils.load_checkpoint(
        RecordingModel,
        checkpoint_path,
        num_samples=20,
        val_num_samples=20,
        context_length=96,
    )

    assert model.kwargs["num_samples"] == 20
    assert model.kwargs["val_num_samples"] == 20
    assert model.kwargs["context_length"] == 96


def test_run_py_forwards_cli_sample_overrides_to_checkpoint_loader():
    source = RUN_PY.read_text()

    assert "num_samples=config_args.model.num_samples" in source
    assert "val_num_samples=getattr(config_args.model, \"val_num_samples\", None)" in source
