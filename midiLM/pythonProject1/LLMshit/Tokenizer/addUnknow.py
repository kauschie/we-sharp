from transformers import GPT2TokenizerFast

for i in range(8):
    tokenizer_path = f"Tokenizers/tokenizer{i}.json"
# 加载分词器
    tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)

    # 检查是否缺少 [UNK] token
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({'unk_token': '[UNK]'})
        print(f"Added [UNK] token to the tokenizer.")

    # 保存更新后的分词器
    tokenizer.save_pretrained(tokenizer_path)