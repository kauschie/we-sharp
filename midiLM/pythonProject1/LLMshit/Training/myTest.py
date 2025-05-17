from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# 初始化预训练模型和分词器
model_path = "../../Model"  # 根据你的路径调整
tokenizer = GPT2TokenizerFast.from_pretrained(model_path, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)

# 输入文本
input_text = ("on_67_0 off_67_240 on_64_0 off_64_240 on_64_0 off_64_240 on_65_0 off_65_240 on_62_0 off_62_240 on_62_0 off_62_240 on_60_0 off_60_240 on_62_0 off_62_240 on_64_0 off_64_240 on_65_0 off_65_240 on_67_0 off_67_240 on_67_0 off_67_240 on_67_0 off_67_240")#
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# gen predict
output_ids = model.generate(
    input_ids,
    max_length=1024,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id  # 处理GPT-2未设置的padding token
)

# 解码生成的token为文本
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Text:")
print(output_text)


