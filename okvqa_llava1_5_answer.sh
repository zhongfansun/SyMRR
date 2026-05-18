#!/bin/bash

gpu_id=0
FILE_PATH=autodl-tmp/output_files/okvqa_answer_self_reflect_generate_val/

CUDA_VISIBLE_DEVICES=$gpu_id swift infer \
--model_type llava1_5-13b-instruct \
--model_cache_dir autodl-tmp/large_VLM_weights/llava-1___5-13b-hf \
--val_dataset autodl-tmp/data_files/input_dataset/okvqa_answer_self_reflect_generate_val.csv \
--result_dir $FILE_PATH

FILE_DIR=$(find -L $FILE_PATH -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
python dataprocess/ans_scores.py $FILE_DIR

