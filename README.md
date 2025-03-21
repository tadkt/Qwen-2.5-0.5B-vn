# Qwen-2.5-0.5B-vn
Finetuning a small foundation model for Vietnamese

# Instruction
Swift does not natively support extra vocabulary, so we refer to swift.llm.model.register to add it manually:

In [line 184][https://github.com/modelscope/ms-swift/blob/74bc60b54b37d138203007b80bea957bc1dd14f3/swift/llm/model/register.py#L184]:
```
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
```
Add another argument extra_vocab_file and set to your extra vocab file:
```
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, extra_vocab_file='Expand_vocab/qwen_extra.tiktoken')
```

# To-do list:

- [ ] Collect Vietnamese dataset: bkai-foundation-models/vi-alpaca, 5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca, https://github.com/telexyz/GPT4VN/tree/main/data
- [ ] Collect English dataset (10-20% compare to Vietnamese only)
- [ ] Transform to the right format in data/sample_data.jsonl (QUANG)
- [ ] Extend vocab for Qwen (read https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md) using pre-extracted bag-of-word (download https://huggingface.co/vinai/phobert-base/blob/main/vocab.txt)
- [ ] Finetune using ms-swift (add dataset, metrics into training script/ adjust hyperparams such as batch_size, learning rate for sample data first/ evaluation VMLU, # add more)
- [ ] More works? Maybe distillation using GPT-4o or GPT-4o mini