import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'model_training/rigweave/scripts'))
from predicted_reference_protocol import validate_config
from hardcase_checkpoint_selection_protocol import assess_candidate

class SelectionTests(unittest.TestCase):
    def config(self,total):
        return json.loads((ROOT/f'model_training/experiments/hardcase{total}k_visual12_20261102.json').read_text())

    def test_both_candidates_are_frozen_zero_update_inherited_only(self):
        for total in (380,390):
            config=self.config(total)
            for mode in ('audit','capacity'): validate_config(config,mode)
            with self.assertRaises(ValueError): validate_config(config,'train')
            for mutate in (
                lambda c:c['checkpoints']['stage1'].update(sha256='0'*64),
                lambda c:c['checkpoint_selection'].update(total_visits=370000),
                lambda c:c['official_smoke'].update(full_beam_capacity_first=True),
                lambda c:c['checkpoint_selection']['manifest'].update(sha256='0'*64),
                lambda c:c['source_deployment'].update(receipt='old_receipt.json')):
                changed=copy.deepcopy(config); mutate(changed)
                with self.assertRaises(ValueError): validate_config(changed,'capacity')

    def test_small_changes_fall_back_but_clear_gain_can_pass(self):
        manifest=json.loads((ROOT/'model_training/selections/hardcase_330k400k_20261102.json').read_text())
        rule=manifest['decision_rule']; base={'edge_f1':.75,'normalized_j2j_successful':.02,'failures':0}
        def check(f1,j2j,losses=None,refs=12,bg=0,failures=0):
            return assess_candidate(base,{'edge_f1':f1,'normalized_j2j_successful':j2j,'failures':failures},losses or [0]*12,refs,bg,rule)
        self.assertFalse(check(.756,.0196)['qualifies'])
        self.assertTrue(check(.77,.0201)['qualifies'])
        self.assertTrue(check(.751,.0189)['qualifies'])
        self.assertFalse(check(.78,.019,losses=[.011]+[0]*11)['qualifies'])
        self.assertFalse(check(.78,.019,refs=11,failures=7)['qualifies'])
        self.assertFalse(check(.78,.019,bg=.006)['qualifies'])

if __name__=='__main__': unittest.main()
