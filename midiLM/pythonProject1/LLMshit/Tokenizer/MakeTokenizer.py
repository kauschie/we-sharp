from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import json


with open(f"../CodeBook/CodeBooks/NBcodebook.json", "r") as f:
    codebook = json.load(f)

# 创建词汇表
vocab = {token: int(idx) for token, idx in codebook.items()}

# 创建分词器
tokenizer = Tokenizer(WordLevel(vocab=vocab))
tokenizer.pre_tokenizer = Whitespace()

# 保存为 tokenizer.json
tokenizer.save(f"Tokenizers/NBtokenizer.json")