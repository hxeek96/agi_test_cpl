#! /bin/bash
BASE_DIR="/home/hs/farnn/memory/experiments"
HOME_DIR="/home/hs/farnn/memory"
REMOTE_DIR="/storage1/hs"
CONFIG_DIR="/home/hs/farnn/memory/config/train"

# Number of Keys
n_keys=1024

# CUDA 설정
mkdir -p "$HOME/lib"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 -m apps.main.train \
    config=${CONFIG_DIR}/hybrid.yaml \
    model.productkey_args.mem_n_keys=${n_keys} \
    model.productkey_args.mem_share_values=true \
    dump_dir=${REMOTE_DIR}/result/hybrid/${n_keys} \
    steps=1000 \
