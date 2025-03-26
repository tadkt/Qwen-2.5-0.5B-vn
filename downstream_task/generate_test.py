# pip install datasets transformers safetensors torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tadkt/Qwen-2.5-0.5B-VN"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "Giới thiệu về nước Việt Nam"
messages = [
    {"role": "system", "content": "Bạn là một trợ lý AI thông minh và hữu ích. Hãy cung cấp câu trả lời chính xác, rõ ràng và dễ hiểu."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_token_type_ids=False, return_tensors="pt").to(model.device)

# Generate the output
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    return_dict_in_generate=False  # Ensure we get token IDs directly
)

# Decode the full output first
full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

# Find the part of the output that corresponds to the assistant's response
# This assumes the assistant's response starts after the user's prompt and system message
input_text_length = len(text)
response_start = full_output.find(text) + input_text_length
response = full_output[response_start:].split("<|im_end|>")[0]

print("Generated Response:", response)