#!/bin/bash
# Qwen3-32B Local Distillation Script
# Uses Qwen3-32B model for distillation (non-MoE version)
# Optimized for 2x A800 GPUs

set -x

# Activate modelscope environment
CONDA_BASE="/baai-cwm-vepfs/cwm/yue.yu/miniconda3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_BASE/envs/modelscope"
else
    echo "ERROR: Conda not found at $CONDA_BASE"
    exit 1
fi

# Show environment info
echo "============================================"
echo "Environment: $(python -c 'import sys; print(sys.executable)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
echo "============================================"

# Check GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
echo "============================================"

# Configuration for Qwen3-32B
# Qwen3-32B is not MoE, so it uses all 32B parameters
# Memory requirement: ~64GB for model + KV cache
GPU_MEMORY_UTIL=0.80  # Use 80% of GPU (~64GB)
BATCH_SIZE=4          # Smaller batch size for 32B model
MAX_TOKENS=4096
TEMPERATURE=0.6
MODEL_PATH="/baai-cwm-vepfs/cwm/yue.yu/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

# Run distillation with Qwen3-32B
CUDA_VISIBLE_DEVICES=0,1 python data_preparation/qwen_local_distill_offline.py \
    --cases_csv data/processed/medqa_train_sft.csv \
    --save_dir data/distil/medqa/qwen3-32b \
    --model_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --tensor_parallel_size 2

echo "============================================"
echo "Qwen3-32B Distillation complete!"
echo "Output: data/distil/medqa/qwen3-32b"
echo "Next step: Run ./scripts/data_preparation/distil_sft_qwen3_32b.sh"
echo "============================================"
