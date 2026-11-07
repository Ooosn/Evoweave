import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'model_training/rigweave/scripts'))
from evaluate_tokenrig_visual12 import CONTINUATION_PHASE, validate_config
from virtual_pair_guard import validate_stage


class ContinuationVisualTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / 'model_training/experiments/tokenrig_native_visual12_20261107.json').read_text())
        validate_config(self.config)
        self.config['execution']['phase'] = CONTINUATION_PHASE
        curve = [{'visits': n, 'stage1_edge_F1': .8, 'GT_MPJPE': .004,
            'checkpoint': {'path': f'/checkpoint_{n}.pt', 'sha256': str(n)}}
            for n in range(310000, 400001, 10000)]
        curve[1]['stage1_edge_F1'] = curve[3]['stage1_edge_F1'] = .84
        curve[3]['GT_MPJPE'] = .003
        self.candidates = [{'label': f"native{r['visits']//1000}k",
            'visits': r['visits'], **r['checkpoint']} for r in (curve[3], curve[1])]
        self.report = {'complete': True, 'passed': True, 'assets': 400000, 'optimizer_updates': 8360,
            'same_parent_panel_input_and_GT': True, 'all_ten_recoveries_retained': True,
            'formal_results_not_used_for_selection': True,
            'rule': 'descending_greedy_F1_then_lower_GT_MPJPE_then_earlier_visits',
            'curve': curve, 'selected_candidates': self.candidates}
        self.config['candidates'] = copy.deepcopy(self.candidates)
        self.write_screening()

    def write_screening(self):
        path = self.root / 'screening.json'
        data = json.dumps(self.report).encode()
        path.write_bytes(data)
        self.config['screening'] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}

    def test_fixed_ranking_tie_break_and_original_policy(self):
        validate_config(self.config)
        self.config['evaluation']['generation_kwargs']['num_beams'] = 1
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_rejects_substituted_endpoint_even_with_new_digest(self):
        self.candidates[0] = {'label': 'native400k', 'visits': 400000, **self.report['curve'][-1]['checkpoint']}
        self.config['candidates'] = copy.deepcopy(self.candidates)
        self.write_screening()
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_audit_and_inference_only_on_exact_admitted_config(self):
        path, output = self.root / 'config.json', self.root / 'out'
        data = json.dumps(self.config).encode()
        path.write_bytes(data)
        state = {'active_operation': {'label': self.config['family'], 'experiment': CONTINUATION_PHASE,
            'explicit_human_authorized': True, 'optimizer_updates_allowed': 0,
            'formal_training_allowed': False, 'gpu_work_allowed': True, 'complete': False,
            'required_entrypoint': 'model_training/rigweave/scripts/evaluate_tokenrig_visual12.py',
            'runs': [{'mode': 'capacity', 'config': str(path), 'output': str(output),
                'config_sha256': hashlib.sha256(data).hexdigest(), 'approved': True}]}}
        args = SimpleNamespace(mode='capacity', config=path, output=output)
        validate_stage(state, self.config, args)
        args.mode = 'train'
        with self.assertRaises(RuntimeError):
            validate_stage(state, self.config, args)


if __name__ == '__main__':
    unittest.main()
