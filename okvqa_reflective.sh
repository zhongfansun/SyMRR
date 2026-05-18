#!/bin/bash

set -e

# 用法:
# bash okvqa_reflective.sh llava1_5 vinvl val 0
# bash okvqa_reflective.sh llava1_6 vinvl val 0

large_model_name=$1
small_model_name=$2
split=$3
gpu_id=${4:-0}

if [ -z "$large_model_name" ] || [ -z "$small_model_name" ] || [ -z "$split" ]; then
  echo "Usage: bash infer.sh <large_model_name> <small_model_name> <split> [gpu_id]"
  echo "Example:"
  echo "  bash infer.sh llava1_5 vinvl val 0"
  echo "  bash infer.sh llava1_6 vinvl val 0"
  exit 1
fi

# 根据 large_model_name 选择模型配置
if [ "$large_model_name" = "llava1_5" ]; then
  model_type="llava1_5-13b-instruct"
  model_cache_dir="autodl-tmp/large_VLM_weights/llava-1___5-13b-hf"
elif [ "$large_model_name" = "llava1_6" ]; then
  model_type="llava1_6-vicuna-13b-instruct"
  # 如果 1.6 的权重目录和 1.5 不一样，改成你自己的实际路径
  model_cache_dir="/root/.cache/modelscope/hub/models/swift/llava-v1.6-vicuna-13b-hf"
else
  echo "Error: unsupported large_model_name: $large_model_name"
  echo "Supported: llava1_5, llava1_6"
  exit 1
fi

FILE_PATH="autodl-tmp/output_files/okvqa_${large_model_name}_${small_model_name}_reflective_${split}/"
VAL_DATASET="autodl-tmp/data_files/input_dataset/okvqa_${small_model_name}_reflective_${split}.csv"

CUDA_VISIBLE_DEVICES=$gpu_id swift infer \
  --model_type "$model_type" \
  --model_cache_dir "$model_cache_dir" \
  --val_dataset "$VAL_DATASET" \
  --result_dir "$FILE_PATH"