#!/bin/bash
# GPU Only Evaluation Script

BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prj_hs


export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"


# CUDA 메모리 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_JIT=0
export TORCH_COMPILE_DISABLE=1

# Checkpoint 경로
CKPT_DIR="${REMOTE_DIR}/result/1024/checkpoints/0000000100"

# Eval 결과 저장 경로
EVAL_DUMP_DIR="${REMOTE_DIR}/eval_results/gpu_$(date +%Y%m%d_%H%M%S)"

cd ${HOME_DIR}

echo "============================================================"
echo "GPU Only Evaluation"
echo "============================================================"
echo "Checkpoint: ${CKPT_DIR}"
echo "Results   : ${EVAL_DUMP_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -m apps.main.eval \
    config=${BASE_DIR}/eval.yaml \
    ckpt_dir=${CKPT_DIR} \
    dump_dir=${EVAL_DUMP_DIR}

echo "============================================================"
echo "Evaluation Complete!"
echo "Results saved to: ${EVAL_DUMP_DIR}"
echo "============================================================"