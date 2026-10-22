#!/usr/bin/env bash
set -Eeuo pipefail

VMAE_HOME="${VMAE_HOME:-/root/gpufree-data/DS_RiboDecode_VMAE_V2_5Fold_ServerBundle_20261022}"
VMAE_PYTHON="${VMAE_PYTHON:-/root/gpufree-data/envs/vmae_v2/bin/python}"
VMAE_RESULTS="${VMAE_RESULTS:-/root/gpufree-data/vmae_v2_results_20261022}"
VMAE_TORCH_URL="${VMAE_TORCH_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.0.0%2Bcu118-cp310-cp310-linux_x86_64.whl}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$VMAE_RESULTS/deployment"
exec 9>"$VMAE_RESULTS/deployment/launch.lock"
flock -n 9 || { echo 'This results directory already has an active training launcher.' >&2; exit 1; }
trap 'code=$?; printf "{\"stage\":\"failed\",\"exit_code\":%d}\n" "$code" > "$VMAE_RESULTS/deployment/status.json"; exit "$code"' ERR
stage() {
  printf '{"stage":"%s","launcher_pid":%d}\n' "$1" "$$" > "$VMAE_RESULTS/deployment/status.json"
  printf '\nVMAE_STAGE=%s\n' "$1"
}

cd "$VMAE_HOME"
stage dependencies
"$VMAE_PYTHON" -m pip install --no-input --no-cache-dir --progress-bar off "$VMAE_TORCH_URL" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$VMAE_PYTHON" -m pip install --no-input --no-cache-dir --progress-bar off -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
"$VMAE_PYTHON" -m pip freeze > "$VMAE_RESULTS/deployment/pip_freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$VMAE_RESULTS/deployment/gpu.csv"
stage validation
"$VMAE_PYTHON" -u run_vmae_v2.py check --gpu
"$VMAE_PYTHON" -m unittest latent_decoder.test_v2 -v
stage benchmark
"$VMAE_PYTHON" -u run_vmae_v2.py benchmark --output "$VMAE_RESULTS"
stage training
"$VMAE_PYTHON" -u run_vmae_v2.py train-all --output "$VMAE_RESULTS"
stage completed
