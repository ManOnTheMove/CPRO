#!/bin/bash
# Qwen3-30B-A3B Local Distillation - Single Script Solution
# No server needed, runs everything in one process
# Optimized for 2x A800 with partial GPU memory usage

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

# Configuration
# gpu_memory_utilization=0.55 means use ~44GB per GPU, leaving room for other processes
# Adjust if you have more/less free GPU memory
GPU_MEMORY_UTIL=0.55
BATCH_SIZE=8
MAX_TOKENS=4096
TEMPERATURE=0.6

# Run distillation (single script, no server needed)
CUDA_VISIBLE_DEVICES=0,1 python data_preparation/qwen_local_distill_offline.py \
    --cases_csv data/processed/medqa_train_sft.csv \
    --save_dir data/distil/medqa/qwen3-local \
    --model_path /baai-cwm-vepfs/cwm/yue.yu/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --tensor_parallel_size 2

echo "============================================"
echo "Distillation complete!"
echo "Next step: Run ./scripts/data_preparation/distil_sft_qwen3.sh"
echo "============================================"
