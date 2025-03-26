# pip install datasets transformers safetensors torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tadkt/Qwen-2.5-0.5B-VN"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
model_inputs = tokenizer([text],  return_token_type_ids=False, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

print("Generated Response:", response)

