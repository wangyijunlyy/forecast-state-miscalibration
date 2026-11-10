import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dacu_backbone_kwargs_are_merged_before_instantiation():
    source = (ROOT / "probts" / "model" / "forecaster" / "prob_forecaster" / "dacu.py").read_text()
    tree = ast.parse(source)
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert "_merge_backbone_kwargs" in function_names
    assert "backbone_kwargs = _merge_backbone_kwargs(self.backbone_init_args, kwargs)" in source
    assert "backbone_cls(**self.backbone_init_args, **kwargs)" not in source
