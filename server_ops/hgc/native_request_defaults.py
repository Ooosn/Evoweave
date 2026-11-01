"""Canonical future Evoweave HGC request defaults; not a dated replay script."""
H_RT_SECONDS = 96*60*60
CPU_SLOTS = 16

def request_argv(keeper_script, source, *, job_name='evoweave-h100'):
    if not str(keeper_script).startswith('/home/wangyy/jobs/') or len(source) != 40:
        raise ValueError('an explicit reviewed keeper script and committed source are required')
    return ['qrsh','-clear','-now','n','-jsv','/geadmin/N1GE/util/resources/jsv/jsv_docker.pl',
        '-q','gpuh.q@pcg01i','-pe','def_slot',str(CPU_SLOTS),'-l','h100=2','-l','mem_req=4G',
        '-l',f'h_rt={H_RT_SECONDS}','-N',job_name,'/bin/bash',str(keeper_script),source]
