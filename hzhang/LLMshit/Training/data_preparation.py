import json
from transformers import PreTrainedTokenizerFast


def prepare_data(tokenizer_path, data_path, output_path):
    # 加载分词器
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)


    # 加载原始 JSON 数据
    with open(data_path, "r") as f:
        data = json.load(f)

    # 转换为 tokenized 格式
    tokenized_texts = [{"input_ids": tokenizer.encode(value)} for value in data.values()]

    # 保存处理后的数据
    with open(output_path, "w") as f:
        json.dump(tokenized_texts, f, indent=2)

    return tokenizer, tokenized_texts