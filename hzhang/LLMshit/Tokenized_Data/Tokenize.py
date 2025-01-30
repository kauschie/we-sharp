import os
import json
from transformers import PreTrainedTokenizerFast

MAX_LENGTH = 750  # 设置最大长度

# 确保输出目录存在
os.makedirs("Train_Datas", exist_ok=True)

for i in range(8):
    # 加载分词器
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"../Tokenizer/Tokenizers/NBtokenizer.json")

    # 显式设置特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "[UNK]"

    # 加载语料
    with open(f"../Yuliao/YuLiaos/yuliao_{i}.json", "r") as f:
        data = json.load(f)

    # 过滤无效值
    data = {key: value for key, value in data.items() if value and isinstance(value, str)}

    # 将文本转为 tokenized 格式
    tokenized_texts = [
        {"input_ids": tokenizer.encode(value, padding="max_length", max_length=MAX_LENGTH)}
        for value in data.values()
    ]

    # 保存处理好的 tokenized 数据
    with open(f"Train_Datas/Tokenized_data{i}.json", "w") as f:
        json.dump(tokenized_texts, f, indent=2)

    print(f"完成第 {i} 批次的分词处理")
