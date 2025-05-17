import os
import json
from transformers import PreTrainedTokenizerFast

MIN_TOKENS = 500  # 过滤低于 500 tokens 的文本

# 确保输出目录存在
os.makedirs("Train_Datas", exist_ok=True)

# 加载 tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"../Tokenizer/Tokenizers/NBtokenizer.json")

# 设置特殊 token（避免报错）
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
if tokenizer.eos_token is None:
    tokenizer.eos_token = "[UNK]"

# 加载语料
with open(f"../Yuliao/YuLiaos/yuliao.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤无效值
data = {key: value for key, value in data.items() if value and isinstance(value, str)}

# 处理文本，删除 token 过短的语料
tokenized_texts = []
for key, value in data.items():
    tokens = tokenizer.encode(value)

    if len(tokens) >= MIN_TOKENS:  # 只保留 token 数量 >= 500 的文本
        tokenized_texts.append({"input_ids": tokens})  # **去掉 max_length 约束**
    else:
        print(f"❌ 过滤掉 {key}: token 数量 {len(tokens)} 低于 {MIN_TOKENS}")

# 保存处理好的 tokenized 数据
output_path = "Train_Datas/Tokenized_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tokenized_texts, f, indent=2)

print(f"✅ 处理完成，共保留 {len(tokenized_texts)} 条数据，存储到 {output_path}")
