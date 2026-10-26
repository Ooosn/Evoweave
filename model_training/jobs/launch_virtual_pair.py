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
    trainer = ('train_reference_xyz.py' if family == 'reference_skeleton_xyz_20261027_v1'
               else 'train_virtual_pair.py')
    main(trainer_module_name='virtual_pair_guard',
         exception_env_name='EVOWEAVE_VIRTUAL_PAIR_SSH_EXCEPTION',
         trainer_script=trainer, modes=('audit', 'capacity', 'train'), cpu_modes=('audit',))
