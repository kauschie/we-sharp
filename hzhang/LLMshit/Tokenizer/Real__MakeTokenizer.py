from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import json
import os

# 加载 codebook
with open("../CodeBook/CodeBooks/NBcodebook.json", "r") as f:
    codebook = json.load(f)

# 添加 UNK token 和 PAD token 到词汇表
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
if UNK_TOKEN not in codebook:
    codebook[UNK_TOKEN] = len(codebook)
if PAD_TOKEN not in codebook:
    codebook[PAD_TOKEN] = len(codebook)

# 创建词汇表
vocab = {token: int(idx) for token, idx in codebook.items()}

# 创建分词器
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
tokenizer.pre_tokenizer = Whitespace()

# 保存分词器为 tokenizer.json
os.makedirs("Tokenizers", exist_ok=True)
tokenizer.save("Tokenizers/NBtokenizer.json")

print(f"分词器已保存到 Tokenizers/NBtokenizer.json")
