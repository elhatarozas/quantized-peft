import os 
import subprocess
from contextlib import contextmanager
import torch 
from accelerate import Accelerator


@contextmanager
def track_memory(write_path):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory: {peak_mem:.2f} GB")
        # with open(write_path+ "/memory.txt", "w+") as fp:
        #    fp.write(f"{peak_mem}")

def main():
    # Initialize accelerator (like harness does)
    accelerator = Accelerator()
    
    print(f"Accelerator device: {accelerator.device}")
    print(f"Distributed type: {accelerator.distributed_type}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Allocate tensor using accelerator's device
    x = torch.randn(10000, 10000).to(accelerator.device)
    
    # Alternatively, if preparing a model/data, harness would do:
    # x = accelerator.prepare(x)
    
    print(f"Tensor device: {x.device}")
    print(f"Tensor shape: {x.shape}")

# with track_memory("2"):

main()