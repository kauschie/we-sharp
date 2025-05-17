import os
import json
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from model_setup import initialize_model
from train_model import train_model

if __name__ == "__main__":
    # === File paths ===
    tokenizer_path = r"..\Tokenizer\Tokenizers\NBtokenizer.json"
    tokenized_data_path = r"..\Yuliao\Yuliaos\yuliao.jsonl"
    output_model_path = r"..\..\Model_768_12_12"
    os.makedirs(output_model_path, exist_ok=True)

    # Step 1: Load the tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    # Step 2: Load tokenized JSONL data line by line
    # Ensure that each `input_ids` is a list of integers
    tokenized_data = []
    with open(tokenized_data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            # Convert string token lists into actual integer lists (if needed)
            if isinstance(entry["input_ids"], str):
                entry["input_ids"] = list(map(int, entry["input_ids"].split()))
            # Use the same input_ids as labels (for language modeling)
            entry["labels"] = entry["input_ids"]
            tokenized_data.append(entry)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(tokenized_data)

    # Step 3: Initialize a Tiny GPT-2 model
    model = initialize_model(tokenizer, model_size="tiny")

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))  # Resize embedding layer to match new vocab

    # Step 4: Train the model
    log_file_path = train_model(model, dataset, tokenizer, output_model_path, run_name="layer_1")

    print(f"✅ Full training log saved to: {log_file_path}")
