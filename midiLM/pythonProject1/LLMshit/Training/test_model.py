from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def test_model(model_path, tokenizer_path, input_text):
    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # 转换输入文本为 token
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 生成文本
    model.eval()
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

    # 解码输出
    return tokenizer.decode(output[0], skip_special_tokens=True)