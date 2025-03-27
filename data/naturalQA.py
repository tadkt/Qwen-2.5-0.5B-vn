import json

# Input and output file names
input_file = '/root/Qwen-2.5-0.5B-vn/NQ-train_pairs_OUTPUT.json'  # Replace with your actual input JSON file name
output_file = '/root/Qwen-2.5-0.5B-vn/NQ-train_pairs_OUTPUT.jsonl'

# System message to be used in all entries
system_content = "Bạn là một trợ lý thông minh. Hãy thực hiện các yêu cầu hoặc trả lời câu hỏi một cách tốt nhất có thể."

try:
    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as json_file:
        input_data = json.load(json_file)
    
    # Open the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in input_data:
            # Create the transformed structure
            transformed = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": item["vi_question"]},
                    {"role": "assistant", "content": item["vi_answer"]}
                ]
            }
            
            # Write each transformed item as a JSON line
            # ensure_ascii=False preserves Vietnamese characters
            json.dump(transformed, outfile, ensure_ascii=False)
            outfile.write('\n')
    
    print(f"Transformation complete. Data saved to '{output_file}'")

except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
except json.JSONDecodeError as e:
    print(f"JSON parsing error in input file: {e}")
    with open(input_file, 'r', encoding='utf-8') as json_file:
        content = json_file.read()
        print("Raw file content:", content)
except Exception as e:
    print(f"An unexpected error occurred: {e}")