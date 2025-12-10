#! /bin/bash
DUMP_DIR="/home/hs/agi_test_cpl/agi_test/test_1/dump"
# Number of Keys
n_keys=8000

# CUDA 설정
mkdir -p "$HOME/lib"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$HOME/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 -m apps.main.train \
    config=/home/hs/agi_test_cpl/agi_test/test_1/config/zero_copy.yaml \
    model.productkey_args.mem_n_keys=${n_keys} \
    model.productkey_args.mem_share_values=true \
    dump_dir=${DUMP_DIR}/${n_keys} \
    steps=100 \
