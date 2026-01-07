#!/bin/bash
# Process Qwen3-32B distilled responses into SFT training format
# Use this after running run_qwen32b_distill.sh

set -x

# Note: Using qwen3-32b instead of deepseek-r1
python data_preparation/distil_sft.py \
    --data_source medqa \
    --split train \
    --data_path data/processed/medqa_train_sft.csv \
    --distil_response_save_dir data/distil/medqa/qwen3-32b \
    --save_path data/verl/medqa_qwen3_32b_distil_sft/train.parquet

python data_preparation/distil_sft.py \
    --data_source medqa \
    --split dev \
    --data_path data/processed/medqa_train_sft.csv \
    --distil_response_save_dir data/distil/medqa/qwen3-32b \
    --save_path data/verl/medqa_qwen3_32b_distil_sft/val.parquet

echo "Done! Output saved to data/verl/medqa_qwen3_32b_distil_sft/"
