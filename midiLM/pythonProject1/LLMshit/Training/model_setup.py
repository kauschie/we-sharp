from transformers import GPT2LMHeadModel, GPT2Config


def initialize_model(tokenizer, model_size="small"):
    """
    初始化 GPT-2 模型，可选择不同模型大小。

    参数:
        tokenizer: 已加载的分词器
        model_size: 选择 "tiny", "small", "medium"（默认为 small）

    返回:
        GPT-2 模型
    """
    # 计算 vocab_size，确保包含所有特殊 token
    vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)

    # 按需调整 vocab_size 为 8 的倍数（如果必须对齐）
    vocab_size = ((vocab_size + 7) // 8) * 8

    # 选择模型规模
    if model_size == "tiny":
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12
        )
    elif model_size == "medium":
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16
        )
    else:  # 默认 small（GPT-2 Small）
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12
        )

    # 初始化模型
    model = GPT2LMHeadModel(config)
    return model
