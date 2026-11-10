import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_forecast_module_uses_validation_sample_override_only_for_val():
    source = (ROOT / "probts" / "model" / "forecast_module.py").read_text()
    tree = ast.parse(source)

    module_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProbTSForecastModule"
    )
    init_func = next(node for node in module_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    init_args = {arg.arg for arg in init_func.args.args}

    assert "val_num_samples" in init_args
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_num_samples_for_stage"
        for node in module_class.body
    )
    assert "self._num_samples_for_stage(stage)" in source
    assert "forecaster.forecast(batch_data, self.num_samples)" in source
