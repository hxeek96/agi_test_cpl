#! /bin/bash
BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"

# Number of Keys
n_keys 8000

# CUDA 설정
mkdir -p "$HOME/lib"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 -m apps.main.train \
    config=${BASE_DIR}/config.yaml \
    model.productkey_args.is_enabled=true \
    model.productkey_args.mem_n_keys=${n_keys} \
    model.productkey_args.mem_share_values=true \
    model.productkey_args.layers=4 \
    model.productkey_args.zero_copy=true \
    model.productkey_args.mem_offload=false \
    model.n_layers=24 \
    dump_dir=${REMOTE_DIR}/result/${n_keys}_zerocopy \
    data.root_dir=${REMOTE_DIR}/data \
    steps=100 \
    profiling.run=false \
    distributed.dp_shard=1 \
    distributed.dp_replicate=1 \
    distributed.tp_size=1 \
    distributed.memory_parallel_size=1
