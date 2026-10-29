#!/usr/bin/env python3
"""Reuse the bounded, whole-process-tree HGC resource supervisor."""
import os
import argparse
import json
from pathlib import Path
import sys

from launch_legacy_single_target import main


def bootstrap_site_environment():
    if os.environ.get('EVOWEAVE_PAIR_SITE_BOOTSTRAPPED') == '1':
        return
    # qrsh command jobs retain JOB_ID/NSLOTS but may omit scheduler tools from PATH.
    script = ('set -e\nsource /home/geadmin/N1GE/default/common/settings.sh\n'
              'source ~/1.sh\nexport EVOWEAVE_PAIR_SITE_BOOTSTRAPPED=1\nexec "$@"')
    os.execv('/bin/bash', ['/bin/bash', '-c', script, 'pair-site-bootstrap',
        sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == '__main__':
    bootstrap_site_environment()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=Path, required=True)
    known, _ = parser.parse_known_args()
    family = json.loads(known.config.read_text()).get('family')
    trainer = {'reference_skeleton_xyz_20261027_v1': 'train_reference_xyz.py',
               'shared_reference_flow_5k_20261027_v1': 'train_reference_flow.py',
               'anchor_joint_binding_5k_20261027_v1': 'train_anchor_joint.py',
               'anchor_integration_5k_20261028_v1': 'train_anchor_integration.py',
               'anchor_full_20261030_v1': 'train_anchor_full.py',
               'ar_only_feedback_5k_20261029_v1': 'train_ar_only.py',
               'stage1_surface_repair_20261029_v1': 'evaluate_stage1_surface_repair.py',
               'predicted_reference_20261029_v1': 'evaluate_predicted_reference.py'}.get(family, 'train_virtual_pair.py')
    raise SystemExit(main(trainer_module_name='virtual_pair_guard',
         exception_env_name='EVOWEAVE_VIRTUAL_PAIR_SSH_EXCEPTION',
         trainer_script=trainer, modes=('audit', 'capacity', 'train'), cpu_modes=('audit',)))
