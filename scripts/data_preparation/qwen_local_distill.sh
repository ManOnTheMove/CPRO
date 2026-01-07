#!/bin/bash
# Run local Qwen3 distillation for Clinical-R1 Cold Start
# Optimized batch processing for 2x A800 80GB

set -x

# Activate modelscope environment (shared path for dev and training machines)
CONDA_BASE="/baai-cwm-vepfs/cwm/yue.yu/miniconda3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_BASE/envs/modelscope"
else
    echo "ERROR: Conda not found. Please activate modelscope manually."
    exit 1
fi

# Configuration
BATCH_SIZE=32          # Concurrent requests per batch (vLLM handles internal batching)
NUM_WORKERS=16         # Parallel worker threads
MAX_TOKENS=4096        # Max output tokens per response
TEMPERATURE=0.6        # Qwen3 recommended for thinking mode
TOP_P=0.95

# Paths
CASES_CSV="data/processed/medqa_train_sft.csv"
SAVE_DIR="data/distil/medqa/qwen3-local"
API_BASE="http://localhost:8000/v1"
MODEL_NAME="/baai-cwm-vepfs/cwm/yue.yu/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"

# Create log directory
mkdir -p logs

echo "============================================"
echo "Qwen3-30B-A3B Local Distillation"
echo "============================================"
echo "Input: $CASES_CSV"
echo "Output: $SAVE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "============================================"

# Check if vLLM server is running
if ! curl -s "$API_BASE/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server is not running at $API_BASE"
    echo "Please start it first with:"
    echo "  ./scripts/data_preparation/start_vllm_server.sh"
    exit 1
fi

echo "vLLM server is running. Starting distillation..."

# Run distillation
python data_preparation/qwen_local_distill.py \
    --cases_csv "$CASES_CSV" \
    --save_dir "$SAVE_DIR" \
    --api_base "$API_BASE" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    2>&1 | tee logs/qwen_distill.log

echo "============================================"
echo "Distillation complete!"
echo "Output saved to: $SAVE_DIR"
echo "============================================"
