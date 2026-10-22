from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_structure_xyz_fill as train
import structure_xyz_large_evaluation as large


def test_trainer_delegates_large_evaluation_without_changing_arguments(monkeypatch):
    config = {'large_evaluation': {'enabled': True}}
    arguments = (object(), object(), object(), config, SimpleNamespace(output=Path('unused')),
                 'cpu', 0, 2, 90000, {'panel': object()}, {'reference': object()}, 'original-small-hash')
    calls = []
    expected = {'complete': True, 'panel_sha256': 'original-small-hash'}
    def evaluate(*args):
        calls.append(args)
        return expected
    monkeypatch.setattr(large, 'run_large_evaluation', evaluate)
    assert train.run_evaluation(*arguments) is expected
    assert calls == [arguments]
