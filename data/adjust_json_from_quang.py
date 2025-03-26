output_file = '/root/Qwen-2.5-0.5B-vn/data/zalo_e2eqa-dpo_adjusted.jsonl'
import json

# Input and output file names
input_file = '/root/Qwen-2.5-0.5B-vn/data/zalo_e2eqa-dpo.jsonl'  # Replace with your actual input JSONL file name

# Process the JSONL file
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Parse each line as JSON
        data = json.loads(line.strip())
        
        # Adjust the assistant response
        assistant_content = data["messages"][2]["content"]  # Access the assistant's content
        if isinstance(assistant_content, list) and len(assistant_content) > 0:
            # If it's a list, take the first item as a string
            data["messages"][2]["content"] = assistant_content[0]
        
        # Write the updated data to the output file
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Transformation complete. Updated data saved to '{output_file}'")