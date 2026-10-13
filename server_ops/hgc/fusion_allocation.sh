#!/usr/bin/env bash
# Reserve the already-authorized two H100s; this script never starts model work.
set -euo pipefail
JOB=/home/wangyy/jobs/virtual_fusion_20261013_v1
export PATH=/home/geadmin/N1GE/bin/lx-amd64:/usr/bin:/bin:$PATH
case "${1:-}" in
    request)
        test "$(hostname -s)" = pcl03
        source ~/1.sh
        test ! -e "$JOB/allocation_job_id_v1.txt"
        test ! -e "$JOB/allocation_requested_v1.txt"
        date -u +%FT%TZ > "$JOB/allocation_requested_v1.txt"
        qstat -u "$USER"
        exec qrsh -clear -now n -pty y \
            -jsv /geadmin/N1GE/util/resources/jsv/jsv_docker.pl \
            -l h100=2 -l mem_req=64G -N virtual_fusion_2h_1013_v1 \
            /bin/bash "$0" hold
        ;;
    hold)
        [[ "${JOB_ID:-}" =~ ^[0-9]+$ ]]
        ulimit -v unlimited
        test "$(ulimit -v)" = unlimited
        printf '%s\n' "$JOB_ID" > "$JOB/allocation_job_id_v1.txt"
        printf '%s\n' "$$" > "$JOB/allocation_keepalive_v1.pid"
        hostname > "$JOB/allocation_host_v1.txt"
        printf '%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" > "$JOB/allocation_cuda_v1.txt"
        printf '%s\n' "${NVIDIA_VISIBLE_DEVICES:-unset}" > "$JOB/allocation_nvidia_v1.txt"
        qstat -j "$JOB_ID" > "$JOB/allocation_qstat_v1.txt"
        nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader \
            > "$JOB/allocation_gpus_v1.csv"
        date -u +%FT%TZ > "$JOB/allocation_ready_v1.txt"
        printf 'FUSION_ALLOCATION_READY job=%s host=%s utc=%s\n' "$JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"
        trap 'rc=$?; printf "%s\n" "$rc" > "$JOB/allocation_keepalive_v1.exit"' EXIT
        trap 'exit 143' TERM
        trap 'exit 129' HUP
        trap 'exit 130' INT
        # Preserve the grant between independently reviewed experiment children.
        while true; do sleep 45; done
        ;;
    *)
        printf 'usage: %s request|hold\n' "$0" >&2
        exit 2
        ;;
esac
