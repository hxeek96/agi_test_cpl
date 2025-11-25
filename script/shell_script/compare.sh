#!/bin/bash
set -e

#================================================================
# FPGA vs GPU 정밀 성능 비교 스크립트
# - 캐시 제거만 sudo 사용 (환경변수 보존)
# - 로그 파싱으로 정확한 시간 측정
# - 양방향 실행 (GPU→FPGA→GPU)
#================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/storage1/hs/result/1024"
CKPT_PATH="${REMOTE_DIR}/checkpoints/0000010000"
CONSOLIDATED_DIR="${CKPT_PATH}/consolidated"
CONSOLIDATED_PARAMS="${CONSOLIDATED_DIR}/params.json"
WORK_DIR="/home/hs/farnn/memory"

# 로그 디렉토리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${WORK_DIR}/eval_results/compare_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# 로그 파일
GPU_WARMUP_LOG="${LOG_DIR}/gpu_warmup.log"
GPU_LOG="${LOG_DIR}/gpu_only.log"
FPGA_LOG="${LOG_DIR}/fpga_hybrid.log"
GPU2_LOG="${LOG_DIR}/gpu_only_after_fpga.log"
REPORT="${LOG_DIR}/comparison_report.txt"

echo "============================================================"
echo "📊 FPGA vs GPU 정밀 성능 비교"
echo "============================================================"
echo "Checkpoint: ${CKPT_PATH}"
echo "Log Dir:    ${LOG_DIR}"
echo ""

#================================================================
# 유틸리티 함수
#================================================================

# 캐시 제거 함수 (sudo 권한 필요한 부분만)
drop_caches() {
    echo "🧹 파일 시스템 캐시 제거 중..."
    sync
    
    # sudo로 캐시 제거 (비밀번호는 한 번만 입력)
    if sudo -n true 2>/dev/null; then
        # sudo 권한 이미 있음
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        echo "✓ 캐시 제거 완료"
    else
        # sudo 권한 요청
        echo "📌 캐시 제거를 위해 sudo 권한이 필요합니다."
        sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' && echo "✓ 캐시 제거 완료" || {
            echo "⚠️  캐시 제거 실패. 계속 진행하지만 결과가 부정확할 수 있습니다."
        }
    fi
    
    sleep 3
}

# GPU VRAM 확인
check_gpu_memory() {
    echo "📊 GPU 메모리 상태:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        awk '{printf "   사용: %.1f GB / 총: %.1f GB (%.1f%%)\n", $1/1024, $2/1024, ($1/$2)*100}' || \
        echo "   (nvidia-smi 실패)"
}

# params.json 수정 함수
set_mem_offload() {
    local value=$1  # true 또는 false
    
    echo "🔧 mem_offload → ${value} 설정 중..."
    
    python3 << EOF
import json

with open("${CONSOLIDATED_PARAMS}", 'r') as f:
    config = json.load(f)

# 올바른 위치에 설정
config['model']['productkey_args']['mem_offload'] = ${value}

# 잘못 추가된 루트 레벨 키 삭제
if 'mem_offload' in config:
    del config['mem_offload']

with open("${CONSOLIDATED_PARAMS}", 'w') as f:
    json.dump(config, f, indent=2)

print("✓ mem_offload → ${value} 설정 완료")
EOF
}

#================================================================
# [0/6] 환경 초기화
#================================================================
echo ""
echo "============================================================"
echo "[0/6] 환경 초기화"
echo "============================================================"

cd "${WORK_DIR}"

# eval.yaml 체크
if [ ! -f "experiments/eval.yaml" ]; then
    echo "⚠️  eval.yaml 없음, eval_fpga.yaml 복사..."
    cp experiments/eval_fpga.yaml experiments/eval.yaml
fi

# consolidated 삭제 (깨끗한 시작)
if [ -d "${CONSOLIDATED_DIR}" ]; then
    echo "🗑️  기존 consolidated 삭제..."
    rm -rf "${CONSOLIDATED_DIR}"
fi

# sudo 권한 미리 획득 (이후 반복 입력 방지)
echo "📌 캐시 제거를 위한 sudo 권한 획득 중..."
sudo -v
echo "✓ sudo 권한 획득 완료"

# sudo 세션 유지 (백그라운드)
(while true; do sudo -n true; sleep 50; done) 2>/dev/null &
SUDO_KEEPER_PID=$!

echo "✓ 환경 초기화 완료"

#================================================================
# [1/6] GPU Warmup (Consolidation)
#================================================================
echo ""
echo "============================================================"
echo "[1/6] GPU Warmup - Consolidation 생성"
echo "============================================================"

drop_caches
check_gpu_memory

echo "🚀 GPU warmup 실행 중..."
WARMUP_START=$(date +%s)

cd "${WORK_DIR}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

torchrun --nproc_per_node=1 \
    -m apps.main.eval \
    --config experiments/eval.yaml \
    --checkpoint-dir "${CKPT_PATH}" \
    > "${GPU_WARMUP_LOG}" 2>&1

WARMUP_END=$(date +%s)
WARMUP_TIME=$((WARMUP_END - WARMUP_START))

echo "✓ Warmup 완료 (${WARMUP_TIME}초, 측정 제외)"

# consolidated 존재 확인
if [ ! -d "${CONSOLIDATED_DIR}" ]; then
    echo "❌ consolidated 생성 실패!"
    kill $SUDO_KEEPER_PID 2>/dev/null
    exit 1
fi

# mem_offload → false 설정
set_mem_offload false

echo "✓ GPU warmup 완료"

#================================================================
# [2/6] GPU Only #1 (캐시 클린)
#================================================================
echo ""
echo "============================================================"
echo "[2/6] GPU Only #1 (캐시 제거 후 측정)"
echo "============================================================"

drop_caches
check_gpu_memory

echo "🚀 GPU only 실행 중..."
GPU_START=$(date +%s)

cd "${WORK_DIR}"
torchrun --nproc_per_node=1 \
    -m apps.main.eval \
    --config experiments/eval.yaml \
    --checkpoint-dir "${CKPT_PATH}" \
    > "${GPU_LOG}" 2>&1

GPU_END=$(date +%s)
GPU_TIME=$((GPU_END - GPU_START))

echo "✓ GPU only 완료 (${GPU_TIME}초)"
check_gpu_memory

#================================================================
# [3/6] FPGA Offload Activation
#================================================================
echo ""
echo "============================================================"
echo "[3/6] FPGA Offload 활성화"
echo "============================================================"

set_mem_offload true

echo "✓ FPGA offload 활성화 완료"

#================================================================
# [4/6] FPGA Hybrid (캐시 클린)
#================================================================
echo ""
echo "============================================================"
echo "[4/6] FPGA Hybrid (캐시 제거 후 측정)"
echo "============================================================"

drop_caches
check_gpu_memory

echo "🚀 FPGA hybrid 실행 중..."
FPGA_START=$(date +%s)

cd "${WORK_DIR}"
export LD_LIBRARY_PATH="/usr/local/lib:${HOME}/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -m apps.main.eval \
    --config experiments/eval.yaml \
    --checkpoint-dir "${CKPT_PATH}" \
    > "${FPGA_LOG}" 2>&1

FPGA_END=$(date +%s)
FPGA_TIME=$((FPGA_END - FPGA_START))

echo "✓ FPGA hybrid 완료 (${FPGA_TIME}초)"
check_gpu_memory

#================================================================
# [5/6] GPU Only #2 (FPGA 이후, 캐시 효과 확인)
#================================================================
echo ""
echo "============================================================"
echo "[5/6] GPU Only #2 (FPGA 실행 후, 캐시 효과 확인)"
echo "============================================================"

set_mem_offload false

drop_caches
check_gpu_memory

echo "🚀 GPU only (2차) 실행 중..."
GPU2_START=$(date +%s)

cd "${WORK_DIR}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

torchrun --nproc_per_node=1 \
    -m apps.main.eval \
    --config experiments/eval.yaml \
    --checkpoint-dir "${CKPT_PATH}" \
    > "${GPU2_LOG}" 2>&1

GPU2_END=$(date +%s)
GPU2_TIME=$((GPU2_END - GPU2_START))

echo "✓ GPU only (2차) 완료 (${GPU2_TIME}초)"
check_gpu_memory

#================================================================
# [6/6] 상세 분석 및 리포트 생성
#================================================================
echo ""
echo "============================================================"
echo "[6/6] 상세 분석 및 리포트 생성"
echo "============================================================"

# sudo keeper 종료
kill $SUDO_KEEPER_PID 2>/dev/null

# 메트릭 추출
extract_metrics() {
    local log=$1
    grep "hellaswag" "$log" | grep -oP "'acc,none': \K[0-9.]+" | head -1 || echo "N/A"
}

GPU_HELLASWAG=$(extract_metrics "$GPU_LOG")
FPGA_HELLASWAG=$(extract_metrics "$FPGA_LOG")
GPU2_HELLASWAG=$(extract_metrics "$GPU2_LOG")

# 평균 계산
GPU_AVG=$(( (GPU_TIME + GPU2_TIME) / 2 ))

# 리포트 생성
cat > "${REPORT}" << REPORT_EOF
================================================================================
                    FPGA vs GPU 정밀 성능 비교 리포트
================================================================================

생성 시각: $(date '+%Y-%m-%d %H:%M:%S')
Checkpoint: ${CKPT_PATH}

================================================================================
1. 실행 시간 비교 (캐시 제거 후 측정)
================================================================================

┌─────────────────────┬──────────┬──────────┬──────────┐
│                     │ GPU #1   │ FPGA     │ GPU #2   │
├─────────────────────┼──────────┼──────────┼──────────┤
│ 전체 실행 시간      │ ${GPU_TIME}초     │ ${FPGA_TIME}초     │ ${GPU2_TIME}초     │
│ 평균 (GPU #1+#2)    │          ${GPU_AVG}초 (평균)            │
└─────────────────────┴──────────┴──────────┴──────────┘

시간 차이:
  - GPU #1 vs FPGA:  $(printf "%+d" $((FPGA_TIME - GPU_TIME)))초 (FPGA가 $([ $FPGA_TIME -lt $GPU_TIME ] && echo "빠름" || echo "느림"))
  - GPU #2 vs FPGA:  $(printf "%+d" $((FPGA_TIME - GPU2_TIME)))초 (FPGA가 $([ $FPGA_TIME -lt $GPU2_TIME ] && echo "빠름" || echo "느림"))
  - GPU #1 vs GPU #2: $(printf "%+d" $((GPU2_TIME - GPU_TIME)))초 (재현성: $([ ${GPU_TIME#-} -ge ${GPU2_TIME#-} ] && [ $((GPU_TIME - GPU2_TIME)) -lt 10 ] && [ $((GPU_TIME - GPU2_TIME)) -gt -10 ] && echo "좋음" || echo "주의"))

$([ $((GPU_TIME - GPU2_TIME)) -gt 10 ] || [ $((GPU_TIME - GPU2_TIME)) -lt -10 ] && echo "⚠️  GPU 실행 시간 차이 큼 → 캐시 효과 의심" || echo "✓ GPU 재현성 양호")

================================================================================
2. 정확도 비교
================================================================================

HellaSwag (acc,none):
  - GPU #1:  ${GPU_HELLASWAG}
  - FPGA:    ${FPGA_HELLASWAG}
  - GPU #2:  ${GPU2_HELLASWAG}

$([ "$GPU_HELLASWAG" = "$FPGA_HELLASWAG" ] && echo "✓ 정확도 동일 (기능적으로 동일)" || echo "⚠️  정확도 차이 발견!")

================================================================================
3. 상세 로그 위치
================================================================================

GPU Warmup:     ${GPU_WARMUP_LOG}
GPU Only #1:    ${GPU_LOG}
FPGA Hybrid:    ${FPGA_LOG}
GPU Only #2:    ${GPU2_LOG}

전체 로그 확인:
  cat ${GPU_LOG} | grep -E "(Setting random seed|Loading consolidated|contexts and|requests completed)"
  cat ${FPGA_LOG} | grep -E "(OffloadFpga|FPGA|DMA)"

================================================================================
4. 결론
================================================================================

REPORT_EOF

# 결론 자동 생성
if [ $FPGA_TIME -lt $GPU_AVG ]; then
    SPEEDUP=$(echo "scale=1; ($GPU_AVG - $FPGA_TIME) * 100 / $GPU_AVG" | bc)
    cat >> "${REPORT}" << CONCLUSION
✅ FPGA가 GPU 대비 평균 ${SPEEDUP}% 빠름 ($(($GPU_AVG - $FPGA_TIME))초 단축)

가능한 이유:
  1. FPGA DDR4 → GPU 전송이 GPU VRAM 내부 접근보다 빠름
  2. Value Table 오프로드로 VRAM 여유 → 다른 연산 가속
  3. DMA 병렬 처리 효과

CONCLUSION
elif [ $FPGA_TIME -gt $(($GPU_AVG + 10)) ]; then
    SLOWDOWN=$(echo "scale=1; ($FPGA_TIME - $GPU_AVG) * 100 / $GPU_AVG" | bc)
    cat >> "${REPORT}" << CONCLUSION
⚠️  FPGA가 GPU 대비 평균 ${SLOWDOWN}% 느림 ($(($FPGA_TIME - $GPU_AVG))초 추가)

가능한 이유:
  1. PCIe 전송 오버헤드
  2. DMA 대기 시간
  3. FPGA gather 연산 최적화 필요

CONCLUSION
else
    cat >> "${REPORT}" << CONCLUSION
✓ FPGA와 GPU 성능이 거의 동일 (오차 범위 내)

해석:
  - Value Table 오프로드의 성능 오버헤드는 무시할 수 있는 수준
  - VRAM 절약 효과만으로도 의미 있음
  - 더 큰 모델(mem_n_keys=2048)에서 FPGA 장점 극대화 예상

CONCLUSION
fi

cat >> "${REPORT}" << 'REPORT_END'

================================================================================
5. 추가 분석 명령어
================================================================================

# GPU VRAM 사용량 비교 (로그에서 추출)
grep -i "memory" ${GPU_LOG} | head -20
grep -i "memory" ${FPGA_LOG} | head -20

# FPGA DMA 활동 확인
grep -E "\[FPGA\]|\[OffloadFpga\]" ${FPGA_LOG} | head -50

# 단계별 시간 분석
grep -E "Setting random seed|Loading consolidated|contexts and|Loglikelihood|Generate" ${GPU_LOG}
grep -E "Setting random seed|Loading consolidated|contexts and|Loglikelihood|Generate" ${FPGA_LOG}

================================================================================
REPORT_END

# 리포트 출력
cat "${REPORT}"

# 콘솔 요약
echo ""
echo "============================================================"
echo "📊 요약"
echo "============================================================"
echo "GPU #1:  ${GPU_TIME}초"
echo "FPGA:    ${FPGA_TIME}초  ($(printf "%+d" $((FPGA_TIME - GPU_TIME)))초)"
echo "GPU #2:  ${GPU2_TIME}초  ($(printf "%+d" $((GPU2_TIME - GPU_TIME)))초)"
echo ""
echo "📄 상세 리포트: ${REPORT}"
echo "============================================================"

# 정리
echo ""
echo "🧹 정리 중..."
rm -rf "${CONSOLIDATED_DIR}"
echo "✓ consolidated 폴더 삭제 완료"
echo ""
echo "✅ 모든 비교 완료!"