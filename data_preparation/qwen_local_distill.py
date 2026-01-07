"""
Qwen3-30B-A3B Local Distillation Script
Replaces DeepSeek API with local vLLM inference for Cold Start data generation.

Optimized for batch processing on 2x A800 80GB GPUs.
"""

import argparse
import os
import asyncio
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="Local Qwen3 distillation for Clinical-R1")
    parser.add_argument("--cases_csv", required=False, 
                        default=os.path.join("data", "processed", "medqa_train_sft.csv"),
                        type=str, help="CSV file with questions")
    parser.add_argument("--save_dir", required=False,
                        default=os.path.join("data", "distil", "medqa", "qwen3-local"),
                        type=str, help="Directory to save distilled responses")
    parser.add_argument("--api_base", required=False,
                        default="http://localhost:8000/v1",
                        type=str, help="vLLM server API base URL")
    parser.add_argument("--model_name", required=False,
                        default="/baai-cwm-vepfs/cwm/yue.yu/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B",
                        type=str, help="Model name/path for vLLM")
    parser.add_argument("--batch_size", required=False, default=16, type=int,
                        help="Batch size for concurrent requests (adjust based on GPU memory)")
    parser.add_argument("--max_tokens", required=False, default=4096, type=int,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", required=False, default=0.6, type=float,
                        help="Sampling temperature (0.6 recommended for thinking mode)")
    parser.add_argument("--top_p", required=False, default=0.95, type=float,
                        help="Top-p sampling parameter")
    parser.add_argument("--num_workers", required=False, default=8, type=int,
                        help="Number of concurrent worker threads")
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
            # Extract thinking content
            think_start = content.index("<think>") + len("<think>")
            think_end = content.index("</think>")
            reasoning = content[think_start:think_end].strip()
            
            # Everything after </think> is the answer
            answer = content[think_end + len("</think>"):].strip()
        except ValueError:
            # Fallback if parsing fails
            reasoning = ""
            answer = content
    
    return answer, reasoning


def get_completion_single(client: OpenAI, prompt: str, model_name: str, 
                          max_tokens: int, temperature: float, top_p: float) -> Tuple[str, str]:
    """
    Get a single completion from Qwen3 via vLLM.
    Returns: (answer, reasoning)
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # Qwen3 thinking mode parameters
            extra_body={
                "top_k": 20,
                "min_p": 0,
            }
        )
        
        content = response.choices[0].message.content
        return parse_qwen3_response(content)
        
    except Exception as e:
        print(f"Error in completion: {e}")
        return None, None


def process_single_question(args_tuple):
    """
    Process a single question. Used for parallel processing.
    """
    client, row, save_dir, model_name, max_tokens, temperature, top_p = args_tuple
    
    question_index = row['question_index']
    output_path = os.path.join(save_dir, f"{question_index}.txt")
    
    # Skip if already processed
    if os.path.exists(output_path):
        return question_index, True, "skipped"
    
    # Prepare prompt
    prompt = "Question: \n" + row["question"]
    if "Your final answer MUST be included in the box \\boxed{}." in prompt:
        prompt = prompt.replace("Your final answer MUST be included in the box \\boxed{}. For example, \\boxed{A} or \\boxed{B}.", "")
    prompt = prompt.strip()
    
    # Get completion
    answer, reasoning = get_completion_single(
        client, prompt, model_name, max_tokens, temperature, top_p
    )
    
    if answer is None:
        return question_index, False, "error"
    
    # Save response
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response_template.format(reasoning=reasoning, response=answer))
        return question_index, True, "success"
    except Exception as e:
        return question_index, False, str(e)


def process_batch_async(client: OpenAI, batch_rows: List[dict], save_dir: str,
                        model_name: str, max_tokens: int, temperature: float, 
                        top_p: float, num_workers: int) -> Tuple[int, int]:
    """
    Process a batch of questions using thread pool for concurrent requests.
    vLLM handles batching internally, so concurrent requests maximize throughput.
    
    Returns: (success_count, skip_count)
    """
    args_list = [
        (client, row, save_dir, model_name, max_tokens, temperature, top_p)
        for row in batch_rows
    ]
    
    success_count = 0
    skip_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_question, args_list))
    
    for question_index, success, status in results:
        if status == "skipped":
            skip_count += 1
        elif success:
            success_count += 1
    
    return success_count, skip_count


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize OpenAI client pointing to local vLLM
    client = OpenAI(
        api_key="EMPTY",  # Not needed for local vLLM
        base_url=args.api_base,
    )
    
    # Test connection
    print(f"Connecting to vLLM server at {args.api_base}...")
    try:
        models = client.models.list()
        print(f"Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM server: {e}")
        print("Please start the vLLM server first with:")
        print(f"  ./scripts/data_preparation/start_vllm_server.sh")
        return
    
    # Load data
    print(f"Loading data from {args.cases_csv}...")
    df = pd.read_csv(args.cases_csv)
    
    # Apply index range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(df)
    df = df.iloc[start_idx:end_idx]
    
    print(f"Processing {len(df)} questions (index {start_idx} to {end_idx})")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Max tokens: {args.max_tokens}")
    print(f"Saving to: {args.save_dir}")
    print("-" * 60)
    
    # Convert to list of dicts for processing
    rows = df.to_dict('records')
    
    # Process in batches
    total_success = 0
    total_skip = 0
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(0, len(rows), args.batch_size), desc="Batches", total=total_batches):
        batch_rows = rows[batch_idx:batch_idx + args.batch_size]
        
        success, skip = process_batch_async(
            client=client,
            batch_rows=batch_rows,
            save_dir=args.save_dir,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_workers=args.num_workers
        )
        
        total_success += success
        total_skip += skip
    
    print("-" * 60)
    print(f"Completed! Success: {total_success}, Skipped: {total_skip}, Failed: {len(rows) - total_success - total_skip}")


if __name__ == '__main__':
    main()
