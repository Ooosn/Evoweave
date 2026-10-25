#!/usr/bin/env python3
"""Reuse the bounded, whole-process-tree HGC resource supervisor."""
from launch_legacy_single_target import main

if __name__ == '__main__':
    main(trainer_module_name='virtual_pair_guard',
         exception_env_name='EVOWEAVE_VIRTUAL_PAIR_SSH_EXCEPTION',
         trainer_script='train_virtual_pair.py', modes=('audit', 'capacity', 'train'), cpu_modes=('audit',))
