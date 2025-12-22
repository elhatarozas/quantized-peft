import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import re 
from peft import PeftModel, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import os 


model_path = "Qwen/Qwen3-1.7B"

model = prepare_model_for_kbit_training(AutoModelForCausalLM(
    model_path
))