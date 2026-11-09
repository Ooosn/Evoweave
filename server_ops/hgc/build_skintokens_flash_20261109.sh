#!/usr/bin/env bash
# SLOTS: 16
set -euo pipefail
[[ "${JOB_ID:-}" == 131942123 && "${NSLOTS:-}" == 16 && "$(hostname -s)" == pcg01 ]]
grep -Eq '^[0-9]+:cpuset:/AGE/131942123\.1/master$' /proc/self/cgroup
grep -Eq '^[0-9]+:devices:/AGE/131942123\.1/master$' /proc/self/cgroup
root=/work/wangyy/skintokens_original_test_20261109
py=$root/venv/bin/python
build=$root/flash_build
affinity=6,7,9,10,11,12,14,15,61,63,64,65,66,67,68,69
umask 077
mkdir "$build"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$build" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME=/opt/cuda/12.4.1 FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=8 NVCC_THREADS=2
sdist=$build/flash_attn-2.8.3.tar.gz
curl --fail --location --silent --show-error --connect-timeout 10 --max-time 120 --output "$sdist" \
  https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
[[ "$(sha256sum "$sdist" | cut -d' ' -f1)" == 1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d ]]
taskset -c "$affinity" "$py" -m pip install --no-cache-dir ninja wheel
taskset -c "$affinity" "$py" -m pip -v wheel --no-deps --no-build-isolation "$sdist" --wheel-dir "$build"
wheel=("$build"/flash_attn-2.8.3-*.whl)
[[ "${#wheel[@]}" == 1 && -f "${wheel[0]}" ]]
sha256sum "${wheel[0]}"
taskset -c "$affinity" "$py" -m pip install --no-deps --force-reinstall "${wheel[0]}"
cd "$root/upstream"
taskset -c "$affinity" "$py" -B demo.py --help
