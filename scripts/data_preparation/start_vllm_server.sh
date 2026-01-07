#!/bin/bash
# Start vLLM server for Qwen3-30B-A3B with 2x A800 GPUs
# Optimized for maximum throughput with batch processing
# REQUIRES: modelscope conda environment with vLLM >= 0.8.0

set -x

# Activate modelscope environment (has transformers 4.57.1 which supports qwen3_moe)
# Use shared conda path that works on both dev and training machines
CONDA_BASE="/baai-cwm-vepfs/cwm/yue.yu/miniconda3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_BASE/envs/modelscope"
else
    echo "ERROR: Conda not found at $CONDA_BASE"
    echo "Please activate modelscope environment manually before running this script:"
    echo "  source /baai-cwm-vepfs/cwm/yue.yu/miniconda3/etc/profile.d/conda.sh"
    echo "  conda activate /baai-cwm-vepfs/cwm/yue.yu/miniconda3/envs/modelscope"
    exit 1
fi

MODEL_PATH="/baai-cwm-vepfs/cwm/yue.yu/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"
PORT=8000
LOG_DIR="$(dirname $0)/../../logs"

# Check if server is already running
if lsof -i:$PORT > /dev/null 2>&1; then
    echo "Port $PORT is already in use. vLLM server may already be running."
    echo "To stop it: kill \$(lsof -t -i:$PORT)"
    exit 1
fi

echo "Starting vLLM server with Qwen3-30B-A3B..."
echo "Model path: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: 2 (for 2x A800)"
echo "Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

# Create log directory
mkdir -p $LOG_DIR

# Start vLLM with optimized settings for 2x A800 80GB
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype bfloat16 \
    2>&1 | tee $LOG_DIR/vllm_server.log

# Parameters explanation:
# --tensor-parallel-size 2: Use 2 GPUs for tensor parallelism
# --max-model-len 8192: Max context length (input + output)
# --gpu-memory-utilization 0.92: Use 92% of GPU memory for KV cache
# --max-num-seqs 64: Max concurrent sequences (batching)
# --max-num-batched-tokens 32768: Max tokens per batch for high throughput
# --enable-chunked-prefill: Better memory efficiency for long sequences
# --dtype bfloat16: Use BF16 for A800 compatibility
