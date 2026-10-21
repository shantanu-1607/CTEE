"""Keep the professor's from-scratch restriction enforceable over all core code."""
import ast
import importlib.util
from pathlib import Path


def test_all_model_and_signal_code_avoids_prebuilt_algorithms():
    root=Path(__file__).resolve().parents[1]
    forbidden={'torch','tensorflow','sklearn','scipy','joblib','jax','keras','autograd'}
    for source in [*root.joinpath('ctee').rglob('*.py'),root/'app.py']:
        tree=ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in forbidden for a in node.names),source
            if isinstance(node,ast.ImportFrom) and node.module:assert node.module.split('.')[0] not in forbidden,source
            if isinstance(node,ast.Call):
                call=ast.unparse(node.func)
                assert not call.startswith(('np.fft.','np.linalg.','np.polyfit','numpy.fft.','numpy.linalg.')), (source,call)


def test_clean_environment_dependencies_are_declared_without_model_frameworks():
    import tomllib
    root=Path(__file__).resolve().parents[1]
    declared=tomllib.loads((root/'pyproject.toml').read_text())['project']['dependencies']
    for name in ['torch','tensorflow','scikit-learn','scipy','joblib']:
        assert not any(requirement.startswith(name) for requirement in declared)
