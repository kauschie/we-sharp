from transformers import GPT2LMHeadModel, GPT2Config

def initialize_model(tokenizer):
    config = GPT2Config(
        vocab_size=((tokenizer.vocab_size+7)//8) * 8,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16
    )
    model = GPT2LMHeadModel(config)
    return model