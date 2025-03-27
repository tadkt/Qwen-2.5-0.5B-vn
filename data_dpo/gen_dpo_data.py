from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Your existing model setup
model_name = "tadkt/Qwen-2.5-0.5B-VN"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Your existing generate_response function
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
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    input_text_length = len(text)
    response_start = full_output.find(text) + input_text_length
    response = full_output[response_start:].split("<|im_end|>")[0]
    return response

# Function to process JSONL file
def process_jsonl(input_file, output_file):
    # Store all processed lines
    processed_lines = []
    
    # Read the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each JSON line
            data = json.loads(line.strip())
            
            # Extract question from user's content
            question = None
            for message in data["messages"]:
                if message["role"] == "user":
                    question = message["content"]
                    break
            
            if question:
                # Generate response
                generated_response = generate_response(question)
                
                # Add rejected_response to the original data
                data["rejected_response"] = generated_response
                processed_lines.append(data)
    
    # Write to new JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Usage
input_file = "/root/Qwen-2.5-0.5B-vn/data_dpo/Vietnamese-Openorca-Multiplechoice_10k_11k.jsonl"  # Replace with your input file path
output_file = "/root/Qwen-2.5-0.5B-vn/data_dpo/Vietnamese-Openorca-Multiplechoice_10k_11k_dpo.jsonl"  # Replace with desired output file path
process_jsonl(input_file, output_file)

print(f"Processing complete. Results written to {output_file}")