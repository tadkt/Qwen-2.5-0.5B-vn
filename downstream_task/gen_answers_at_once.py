from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

model_name = "tadkt/Qwen-2.5-0.5B-VN"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

questions = [
    "Giới thiệu về nước Việt Nam",
    "Thủ đô của Việt Nam là gì?",
    "Món ăn truyền thống của Việt Nam là gì?"
]

def generate_response(question):
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý AI thông minh và hữu ích. Hãy cung cấp câu trả lời chính xác, rõ ràng và dễ hiểu."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_token_type_ids=False, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    # Decode the full output first
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # Find the part of the output that corresponds to the assistant's response
    # This assumes the assistant's response starts after the user's prompt and system message
    input_text_length = len(text)
    response_start = full_output.find(text) + input_text_length
    response = full_output[response_start:].split("<|im_end|>")[0]
    return response

results = []
for question in questions:
    response = generate_response(question)
    results.append({"Question": question, "Response": response})

# Save to CSV file
df = pd.DataFrame(results)
df.to_csv("responses.csv", index=False, encoding="utf-8")

print("Responses saved to responses.csv")
