import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def benchmark_speed(model_path, num_prompts=5, num_tokens=256, num_runs=3):
    """
    Benchmark speed.
    
    Args:
        model_path: HuggingFace model path
        num_prompts: Number of HumanEval prompts to test (default: 5)
        num_tokens: Number of tokens to generate for throughput (default: 256)
        num_runs: Number of runs per prompt for averaging (default: 3)
    
    Returns:
        tuple: (mean_ttft_ms, mean_throughput_tokens_per_sec)
    """
    
    print(f"Loading model: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16, # TODO: Adjust for quantized model/ PEFT Model
        device_map="cuda:0",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get HumanEval prompts
    print(f"Loading {num_prompts} HumanEval prompts...")
    dataset = load_dataset("openai_humaneval", split="test")
    prompts = [dataset[i]["prompt"] for i in range(num_prompts)]
    
    ttft_measurements = []
    throughput_measurements = []
    
    # Warmup run (important for accurate timing)
    print("Warming up...")
    input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    
    print(f"Running benchmark ({num_runs} runs per prompt)...")
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"  Prompt {prompt_idx + 1}/{num_prompts}")
        
        for run in range(num_runs):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            prompt_length = input_ids.shape[1]
            
            # Measure Time to First Token (TTFT)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # Generate first token only
                first_token_output = model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            ttft = time.perf_counter() - start_time
            ttft_measurements.append(ttft * 1000)  # Convert to ms
            
            # Measure Throughput (tokens/sec for generation phase)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                full_output = model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    do_sample=True,  # More realistic than greedy
                    temperature=0.2,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            generation_time = time.perf_counter() - start_time
            
            # Calculate tokens/sec (excluding prompt processing)
            num_generated = full_output.shape[1] - prompt_length
            tokens_per_sec = num_generated / generation_time
            throughput_measurements.append(tokens_per_sec)
    
    # Calculate statistics
    mean_ttft = np.mean(ttft_measurements)
    std_ttft = np.std(ttft_measurements)
    mean_throughput = np.mean(throughput_measurements)
    std_throughput = np.std(throughput_measurements)
    
    print(f"\n{'='*60}")
    print(f"Results for {model_path}")
    print(f"{'='*60}")
    print(f"Time to First Token (TTFT):")
    print(f"  Mean: {mean_ttft:.2f} ms")
    print(f"  Std:  {std_ttft:.2f} ms")
    print(f"\nThroughput (decode phase):")
    print(f"  Mean: {mean_throughput:.2f} tokens/sec")
    print(f"  Std:  {std_throughput:.2f} tokens/sec")
    print(f"{'='*60}\n")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return (mean_ttft, mean_throughput)

if __name__ == "__main__":
    # Example usage
    model_path = "Qwen/Qwen3-1.7B"
    
    ttft, throughput = benchmark_speed(
        model_path,
        num_prompts=5,      # Test on first 5 HumanEval problems
        num_tokens=512,     # Industry standard (matches typical code completion)
        num_runs=3          # 3 runs per prompt = 15 total measurements
    )
    
    print(f"Summary: TTFT={ttft:.2f}ms, Throughput={throughput:.2f} tok/s")