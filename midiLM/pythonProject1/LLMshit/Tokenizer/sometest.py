from transformers import PreTrainedTokenizerFast

# 加载分词器
tokenizer = PreTrainedTokenizerFast(tokenizer_file="Tokenizers/tokenizer0.json")

# 检查 <unk> token
print("UNK token:", tokenizer.unk_token)  # 应输出 "<unk>"

if tokenizer.unk_token is None:
    tokenizer.add_special_tokens({'unk_token': '<unk>'})
    print(f"Added <unk> token to the tokenizer.")

# 验证是否设置成功
print("UNK token:", tokenizer.unk_token)

# 保存更新后的分词器
tokenizer.save_pretrained("Tokenizers/tokenizer0.json")