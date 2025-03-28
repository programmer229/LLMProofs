from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
current_eos_ids = tokenizer.eos_token_id
print(current_eos_ids)
print(tokenizer.eos_token)

# Add the new token 
new_token = "<｜end▁of▁sentence｜>"
tokenizer.add_tokens([new_token], special_tokens=True)

# Get its ID
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print(new_token_id)

# Get current EOS ID(s)
current_eos_ids = tokenizer.eos_token_id
print(current_eos_ids)

