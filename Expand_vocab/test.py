# input_file = "vocab.txt"  
# output_file = "vocab_base64.txt"

# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         parts = line.strip().split(" ", 1)  # Split only at the first space
#         if len(parts) == 2:
#             token, freq = parts
#             outfile.write(f"{token}\t{freq}\n")

# print(f"Converted file saved as {output_file}")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True, extra_vocab_file="qwen_extra.tiktoken")

token = tokenizer("anh dọn dẹp xong anh qua")
print(token)
