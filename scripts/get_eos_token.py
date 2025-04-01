from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
current_eos_ids = tokenizer.eos_token_id
print("current_eos_ids: ", current_eos_ids)
print("tokenizer.eos_token: ", tokenizer.eos_token)

# Add the new token 
new_token = "<|endoftext|>"
tokenizer.add_tokens([new_token], special_tokens=True)

# Get its ID
new_token_id = tokenizer.convert_tokens_to_ids(new_token)
print("new_token_id: ", new_token_id)

# Get current EOS ID(s)
current_eos_ids = tokenizer.eos_token_id
print("current_eos_ids: ", current_eos_ids)
print("eos_token: ", tokenizer.eos_token)

# 151645

tokenizer.eos_token_id = 151643
print("tokenizer.eos_token_id: ", tokenizer.eos_token_id)
print("tokenizer.eos_token: ", tokenizer.eos_token)