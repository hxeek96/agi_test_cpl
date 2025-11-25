#!/bin/bash
# FPGA Offload Evaluation Script

BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prj_hs

# DMA 라이브러리 경로
export LD_LIBRARY_PATH="/home/hs/farnn/memory/host_dma:${LD_LIBRARY_PATH}"
# CUDA 메모리 설정
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
export PYTORCH_JIT=0
export TORCH_COMPILE_DISABLE=1
# Checkpoint 경로 설정 
CKPT_DIR="${REMOTE_DIR}/result/1024/checkpoints/0000000100"
CONSOLIDATED_PARAMS="${CKPT_DIR}/consolidated/params.json"
# Eval 결과 저장 경로
EVAL_DUMP_DIR="${REMOTE_DIR}/eval_results/fpga_$(date +%Y%m%d_%H%M%S)"
cd ${HOME_DIR}
echo "============================================================"
echo "FPGA Offload Evaluation"
echo "============================================================"
echo "Checkpoint: ${CKPT_DIR}"
echo "Results   : ${EVAL_DUMP_DIR}"
echo "============================================================"
# mem_offload → true 설정
echo "🔧 mem_offload → true ..."
if [ -f "${CONSOLIDATED_PARAMS}" ]; then
    python3 << EOF
import json
with open("${CONSOLIDATED_PARAMS}", 'r') as f:
    config = json.load(f)
config['model']['productkey_args']['mem_offload'] = True
if 'mem_offload' in config:
    del config['mem_offload']
with open("${CONSOLIDATED_PARAMS}", 'w') as f:
    json.dump(config, f, indent=2)
print("✓ mem_offload → true 설정 완료")
EOF
else
    echo " No consolidated/params.json."
    exit 1
fi
echo ""
CUDA_VISIBLE_DEVICES=0 python -m apps.main.eval \
    config=${BASE_DIR}/eval_fpga.yaml \
    ckpt_dir=${CKPT_DIR} \
    dump_dir=${EVAL_DUMP_DIR}
echo "============================================================"
echo "Evaluation Complete!"
echo "Results saved to: ${EVAL_DUMP_DIR}"
echo "============================================================"

