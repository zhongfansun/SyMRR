#!/bin/bash


BASE_PATH=$(pwd)
echo "当前工作目录: $BASE_PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

export PYTHONPATH=/root/ms-swift-main/custom_lib:$PYTHONPATH

gpu_id=0

TARGET_PATH="output/llava1_5-13b-instruct"
echo "目标工作目录: $TARGET_PATH"

wait
sync



NEWEST_DIR="output/llava1_5-13b-instruct/v28-20260503-155723"

CUDA_VISIBLE_DEVICES=$gpu_id swift infer --ckpt_dir "$NEWEST_DIR/checkpoint-563-merged" --load_dataset_config true

