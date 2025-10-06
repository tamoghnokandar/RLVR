import argparse
import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


base_model_name = "Qwen/Qwen3-1.7B"

base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
    )




adapter_model_name_or_path = f"outputs/checkpoint-146"
merged_model_name_or_path = f"outputs-merged"

print("Merged model will be saved to: ", merged_model_name_or_path)



model = PeftModel.from_pretrained(base_model, adapter_model_name_or_path)
model = model.merge_and_unload()

model.save_pretrained(merged_model_name_or_path)
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_model_name_or_path)