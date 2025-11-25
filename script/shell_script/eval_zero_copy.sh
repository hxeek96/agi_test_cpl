#!/bin/bash
# Zero-Copy Evaluation Script

set -euo pipefail

BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"
CKPT_STEP="0000000100"

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prj_hs

# 라이브러리 경로
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# CUDA 메모리 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_JIT=0
export TORCH_COMPILE_DISABLE=1

CKPT_DIR="${REMOTE_DIR}/result/zero_copy/1024/checkpoints/${CKPT_STEP}"
EVAL_DUMP_DIR="${REMOTE_DIR}/eval_results/zero_copy_$(date +%Y%m%d_%H%M%S)"

cd "${HOME_DIR}"

echo "============================================================"
echo " Zero-Copy Evaluation"
echo "------------------------------------------------------------"
echo " Checkpoint : ${CKPT_DIR}"
echo " Dump Dir   : ${EVAL_DUMP_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -m apps.main.eval \
    config="${BASE_DIR}/eval_zero_copy.yaml" \
    ckpt_dir="${CKPT_DIR}" \
    dump_dir="${EVAL_DUMP_DIR}"

echo "============================================================"
echo " Evaluation Complete!"
echo " Results saved to: ${EVAL_DUMP_DIR}"
echo "============================================================"
