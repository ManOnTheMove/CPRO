"""
Qwen3-30B-A3B Local Distillation Script (Offline Mode)
Uses vLLM's Python API directly - no server needed, single script solution.

Optimized for batch processing on 2x A800 80GB GPUs.
"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# vLLM offline inference
from vllm import LLM, SamplingParams


response_template = """<think>
{reasoning}
</think>
<answer>
{response}
</answer>"""

system_prompt = """You are a helpful medical assistant. Answer the following multiple choice question. 
Think step by step and show your reasoning process.
Your final answer MUST be included in the box \\boxed{}. For example, \\boxed{A} or \\boxed{B}."""


def parse_args():
    parser = argparse.ArgumentParser(description="Local Qwen3 distillation (offline mode)")
    parser.add_argument("--cases_csv", required=False, 
                        default=os.path.join("data", "processed", "medqa_train_sft.csv"),
                        type=str, help="CSV file with questions")
    parser.add_argument("--save_dir", required=False,
                        default=os.path.join("data", "distil", "medqa", "qwen3-local"),
                        type=str, help="Directory to save distilled responses")
    parser.add_argument("--model_path", required=False,
                        default="/baai-cwm-vepfs/cwm/yue.yu/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B",
                        type=str, help="Model path")
    parser.add_argument("--batch_size", required=False, default=8, type=int,
                        help="Batch size for inference")
    parser.add_argument("--max_tokens", required=False, default=4096, type=int,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", required=False, default=0.6, type=float,
                        help="Sampling temperature")
    parser.add_argument("--top_p", required=False, default=0.95, type=float,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", required=False, default=20, type=int,
                        help="Top-k sampling parameter")
    parser.add_argument("--gpu_memory_utilization", required=False, default=0.55, type=float,
                        help="GPU memory utilization (0.55 for ~48GB free)")
    parser.add_argument("--tensor_parallel_size", required=False, default=2, type=int,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--start_idx", required=False, default=0, type=int,
                        help="Starting index (for resuming)")
    parser.add_argument("--end_idx", required=False, default=-1, type=int,
                        help="Ending index (-1 for all)")
    args = parser.parse_args()
    return args


def parse_qwen3_response(content: str) -> Tuple[str, str]:
    """
    Parse Qwen3 thinking mode response.
    Extracts reasoning from <think>...</think> and the rest as answer.
    
    Returns: (answer, reasoning)
    """
    reasoning = ""
    answer = content
    
    if "<think>" in content and "</think>" in content:
        try:
            think_start = content.index("<think>") + len("<think>")
            think_end = content.index("</think>")
            reasoning = content[think_start:think_end].strip()
            answer = content[think_end + len("</think>"):].strip()
        except ValueError:
            reasoning = ""
            answer = content
    
    return answer, reasoning


def build_prompt(question: str) -> str:
    """Build the full prompt with system message and user question."""
    # Clean up the question
    prompt = "Question: \n" + question
    if "Your final answer MUST be included in the box \\boxed{}." in prompt:
        prompt = prompt.replace("Your final answer MUST be included in the box \\boxed{}. For example, \\boxed{A} or \\boxed{B}.", "")
    prompt = prompt.strip()
    
    # Format as chat message
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return full_prompt


def process_batch(llm: LLM, batch_rows: List[dict], save_dir: str, 
                  sampling_params: SamplingParams) -> Tuple[int, int]:
    """
    Process a batch of questions using vLLM's batch inference.
    Returns: (success_count, skip_count)
    """
    # Filter out already processed questions
    to_process = []
    skip_count = 0
    
    for row in batch_rows:
        output_path = os.path.join(save_dir, f"{row['question_index']}.txt")
        if os.path.exists(output_path):
            skip_count += 1
        else:
            to_process.append(row)
    
    if not to_process:
        return 0, skip_count
    
    # Build prompts
    prompts = [build_prompt(row["question"]) for row in to_process]
    
    # Generate responses in batch
    outputs = llm.generate(prompts, sampling_params)
    
    # Save results
    success_count = 0
    for row, output in zip(to_process, outputs):
        try:
            generated_text = output.outputs[0].text
            answer, reasoning = parse_qwen3_response(generated_text)
            
            output_path = os.path.join(save_dir, f"{row['question_index']}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_template.format(reasoning=reasoning, response=answer))
            success_count += 1
        except Exception as e:
            print(f"Error processing question {row['question_index']}: {e}")
    
    return success_count, skip_count


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Qwen3-30B-A3B Local Distillation (Offline Mode)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.cases_csv}")
    print(f"Output: {args.save_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 60)
    
    # Initialize vLLM with offline mode
    print("\nLoading model (this may take a few minutes)...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    print("Model loaded successfully!")
    
    # Set sampling parameters (Qwen3 recommended settings)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )
    
    # Load data
    print(f"\nLoading data from {args.cases_csv}...")
    df = pd.read_csv(args.cases_csv)
    
    # Apply index range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(df)
    df = df.iloc[start_idx:end_idx]
    
    print(f"Processing {len(df)} questions (index {start_idx} to {end_idx})")
    
    # Convert to list of dicts
    rows = df.to_dict('records')
    
    # Process in batches
    total_success = 0
    total_skip = 0
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(0, len(rows), args.batch_size), desc="Processing", total=total_batches):
        batch_rows = rows[batch_idx:batch_idx + args.batch_size]
        success, skip = process_batch(llm, batch_rows, args.save_dir, sampling_params)
        total_success += success
        total_skip += skip
    
    print("=" * 60)
    print(f"Completed!")
    print(f"  Success: {total_success}")
    print(f"  Skipped (already exist): {total_skip}")
    print(f"  Failed: {len(rows) - total_success - total_skip}")
    print(f"  Output: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
