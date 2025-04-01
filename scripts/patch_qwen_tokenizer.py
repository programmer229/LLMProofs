from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
import os

"""
Run this script once to fix the EOS token used by the Qwen2 modles downloaded on your machine.
Modify the models you patch below.

You only need to run this script once after you download a model and then you're good forever, unless you re-download a different version of the model.
"""

def patch_qwen2_eos_token(model_name):
    cache_file = hf_hub_download(repo_id=model_name, filename="config.json")
    cache_dir = os.path.dirname(cache_file)

    # Fix tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.save_pretrained(cache_dir)

    # Fix model config
    config = AutoConfig.from_pretrained(model_name)
    config.eos_token_id = tokenizer.eos_token_id
    config.save_pretrained(cache_dir)

if __name__ == "__main__":
    patch_qwen2_eos_token("Qwen/Qwen2.5-1.5B-Instruct")
    patch_qwen2_eos_token("Qwen/Qwen2.5-7B-Instruct")
    patch_qwen2_eos_token("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")