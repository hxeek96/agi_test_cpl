#! /bin/bash
BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"
# CKPT_PATH=

# Number of Keys
n_keys=1024
# libcuda.so를 사용자 홈에서 노출 (root 필요 없음)
mkdir -p "$HOME/lib"
# ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.535.230.02 "$HOME/lib/libcuda.so"
# export LIBRARY_PATH="$HOME/lib:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH}"
# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"

# ⚠️ FPGA P2P를 위한 PyTorch 메모리 allocator 설정
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False

# DMA 라이브러리 경로
export LD_LIBRARY_PATH="/home/hs/farnn/memory/host_dma:${LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 -m apps.main.train \
    config=${BASE_DIR}/pkplus_373m_1024k.yaml \
    model.productkey_args.is_enabled=true \
    model.productkey_args.mem_offload=true \
    model.productkey_args.mem_n_keys=${n_keys} \
    model.productkey_args.mem_share_values=true \
    model.productkey_args.layers=4 \
    model.n_layers=24 \
    dump_dir=${REMOTE_DIR}/result \
    data.root_dir=${REMOTE_DIR}/data \
    data.batch_size=1 \
    data.seq_len=2 \
    steps=10 \
    profiling.run=false \
    distributed.dp_shard=1 \
    distributed.dp_replicate=1 \
    distributed.tp_size=1 \
    distributed.memory_parallel_size=1

