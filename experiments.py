import os 
import subprocess
from contextlib import contextmanager
import torch 
import sys 
from main import main as harness_main
from extraeval import benchmark_speed
import pandas as pd 

local_model_path = "./models"

defaults_harness_settings = [
    "--tasks", "humaneval",
        "--precision", "bf16",
        "--max_length_generation", "512",
        "--temperature", "0.2",
        "--batch_size", "1",
        "--allow_code_execution",
        "--n_samples",  "1",
        "--save_generations", 
        "--save_references"
]

experiment_dict = {
        "base_qwen_bf16" : [
        "--model", "Qwen/Qwen3-1.7B",
        ], 
        "nf4_qwen_bf16" : [
        "--model", local_model_path + "/qwennf4" # TODO: Inspect harness for quantization support + Inspect extra eval for quantization compatbility
        ]
    }

@contextmanager
def track_memory(write_path):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(peak_mem)
        with open(write_path+ "/memory.txt", "w+") as fp:
            fp.write(f"{peak_mem}")

def main_func():
    print(torch.cuda.is_available()) 
    base_path = "results"
    full_experiment_dict = {experiment : settings + defaults_harness_settings  for experiment, settings in experiment_dict.items()}
    for experiment_name, predefined_args in full_experiment_dict.items():
        write_path = base_path + "/" + experiment_name 
        if not os.path.exists(write_path): 
            os.mkdir(write_path)
        args = predefined_args + ["--metric_output_path", write_path + "/evaluation_results.json", 
                                  "--save_generations_path", write_path + "/generations.json"
                                  ]
        sys.argv = ["main.py"] + args 
        print("------------------------------------\n")
        print(f"Running the code evaluation on {experiment_name}")
        
        harness_main()
        print("Running Time and Memory benchmark:")
        with track_memory(write_path):
            ttft, ts = benchmark_speed(args[1]) # presuming the model name will always be at the same place (not ideal but this works for now)
        
        pd.Series([ttft, ts], index=["ttft", "t/s"]).to_csv(write_path + "/speed.csv")

if __name__ == "__main__": 
    main_func()