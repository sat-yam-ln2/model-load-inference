import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# download model and tokenizer

model_name = "Qwen/Qwen2-Math-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"cache/tokenizer/{model_name}")


# download and save the model

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(f"cache/model/{model_name}")
