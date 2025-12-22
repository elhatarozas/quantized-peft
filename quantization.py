from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

model_id = "Qwen/Qwen3-1.7B"
# simple QLoRA NF4 quant to compare
quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True, 
                                         bnb_4bit_quant_type="nf4", 
                                         bnb_4bit_compute_dtype=torch.bfloat16,
                                         bnb_4bit_use_double_quant=True
                                         )

tokenizer = AutoTokenizer.from_pretrained(
    model_id  # Original base model
)
# tokenizer.save_pretrained("models/qwennf4")
# model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", quantization_config=quantization_config)
# model_nf4.save_pretrained("models/qwennf4")
model_nf4 = AutoModelForCausalLM.from_pretrained("models/qwennf4",  
                                                trust_remote_code=True, 
                                                device_map="cuda:0")
# tokenizer = AutoTokenizer.from_pretrained("models/qwennf4")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Who is Super Mario?"
input = tokenizer(prompt, return_tensors="pt").input_ids.to(model_nf4.device)
print(tokenizer.decode(model_nf4.generate(input, max_new_tokens=128)[0], skip_special_tokens=True))
# tokenizer

