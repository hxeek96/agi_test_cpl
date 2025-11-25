#!/bin/bash
# Checkpoint Consolidation Script
# FSDP 분산 체크포인트를 단일 consolidated.pth로 변환

BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prj_hs

# CUDA 라이브러리 경로
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Checkpoint 경로 설정
CKPT_DIR="${REMOTE_DIR}/result/1024/checkpoints/0000000100"
CONSOLIDATED_DIR="${CKPT_DIR}/consolidated"
CONSOLIDATED_PARAMS="${CONSOLIDATED_DIR}/params.json"

cd ${HOME_DIR}

echo "============================================================"
echo "Checkpoint Consolidation"
echo "============================================================"
echo "Checkpoint: ${CKPT_DIR}"
echo "Target:     ${CONSOLIDATED_DIR}"
echo "============================================================"

# 기존 consolidated 삭제 (선택 사항)
if [ -d "${CONSOLIDATED_DIR}" ]; then
    echo "⚠️  기존 consolidated 폴더가 있습니다."
    read -p "삭제하고 다시 생성하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  기존 consolidated 삭제 중..."
        rm -rf "${CONSOLIDATED_DIR}"
    else
        echo "ℹ️  기존 consolidated 폴더를 유지합니다."
        exit 0
    fi
fi

echo ""
echo "🔄 Consolidation 실행 중..."
echo ""

# Python으로 consolidation 수행
python3 << EOF
import sys
import os
sys.path.insert(0, "${HOME_DIR}")

from pathlib import Path
from apps.main.eval import consolidate_checkpoints

checkpoint_dir = Path("${CKPT_DIR}")
consolidate_path = checkpoint_dir / "consolidated"

print(f"📦 Consolidating checkpoint...")
print(f"   Source: {checkpoint_dir}")
print(f"   Target: {consolidate_path}")

try:
    consolidate_checkpoints(checkpoint_dir, consolidate_path)
    print("✓ Consolidation 완료!")
except Exception as e:
    print(f"❌ Consolidation 실패: {e}")
    sys.exit(1)
EOF

# Consolidation 성공 확인
if [ ! -d "${CONSOLIDATED_DIR}" ]; then
    echo "❌ Consolidation 실패!"
    exit 1
fi

echo ""
echo "🔧 mem_offload → false 설정 중..."

# mem_offload를 false로 초기화
if [ -f "${CONSOLIDATED_PARAMS}" ]; then
    python3 << EOF
import json

with open("${CONSOLIDATED_PARAMS}", 'r') as f:
    config = json.load(f)

# mem_offload를 false로 설정 (GPU 기본값)
config['model']['productkey_args']['mem_offload'] = False

# 잘못 추가된 루트 레벨 키 삭제
if 'mem_offload' in config:
    del config['mem_offload']

with open("${CONSOLIDATED_PARAMS}", 'w') as f:
    json.dump(config, f, indent=2)

print("✓ mem_offload → false 설정 완료")
EOF
else
    echo "⚠️  params.json이 생성되지 않았습니다."
fi

echo ""
echo "============================================================"
echo "✅ Consolidation 완료!"
echo "============================================================"
echo "Consolidated 경로: ${CONSOLIDATED_DIR}"
echo ""
echo "다음 단계:"
echo "  1. GPU 평가:  bash eval.sh"
echo "  2. FPGA 평가: bash eval_fpga.sh"
echo "============================================================"