import json
from transformers import PreTrainedTokenizerFast
from model_setup import initialize_model
from train_model import train_model
# from test_model import test_model

if __name__ == "__main__":
    # 文件路径
    tokenizer_path = "../Tokenizer/Tokenizers/NBtokenizer.json"
    tokenized_data_path = "../Tokenized_Data/Train_Datas/Tokenized_data0.json"
    output_model_path = r"C:\Users\HuaiyuZ\PycharmProjects\We_sharp_Jan20\LLMshit\Models\Model0"

    # Step 1: 加载分词器
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # 加载已生成的 tokenized_data
    with open(tokenized_data_path, "r") as f:
        tokenized_data = json.load(f)

    # 为每个数据添加 labels
    for data in tokenized_data:
        data["labels"] = data["input_ids"]

    # Step 2: 初始化模型
    model = initialize_model(tokenizer)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))  # Adjust model vocab size

    # Step 3: 训练模型
    log_file_path = train_model(model, tokenized_data, tokenizer, output_model_path, "layer_1")

    print(f"完整日志文件已保存：{log_file_path}")

    # Step 4: 测试模型（可选）
    # input_text = "3880531 3880763 8040296"
    # result = test_model(output_model_path, output_model_path, input_text)
    # print("Generated Output:", result)
