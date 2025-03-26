import json

# Load the JSON file
input_file = '/root/Qwen-2.5-0.5B-vn/data/mlqa-dpo.json'  # Replace with your actual input file name
with open(input_file, 'r', encoding='utf-8') as json_file:
    input_data = json.load(json_file)

# Open a file to write JSONL
output_file = '/root/Qwen-2.5-0.5B-vn/data/mlqa-dpo_qwen-vn.jsonl'
with open(output_file, 'w', encoding='utf-8') as outfile:
    for item in input_data:
        # Create the new structure
        try:
            transformed = {
                "messages": [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["chosen"][0]}  # Take first item from chosen list
                ],
                "rejected_response": item['rejected'][0]
            }
        except:
            continue
        # Write each transformed item as a JSON line
        json.dump(transformed, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Transformation complete. Data saved to '{output_file}'")