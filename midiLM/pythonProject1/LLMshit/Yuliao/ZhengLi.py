import os
import json
from transformers import PreTrainedTokenizerFast

def collect_files_to_jsonl(folder_path, output_jsonl_path, tokenizer):
    """
    Traverse all .txt files in a folder, tokenize their content, and save each as a line in a JSONL file.
    Each line in the JSONL file is a JSON object with the format: {"input_ids": [token1, token2, ...]}.
    This format is suitable for training GPT-2 and similar models.

    Args:
        folder_path (str): Path to the folder containing input .txt files
        output_jsonl_path (str): Path to the output JSONL file
        tokenizer (PreTrainedTokenizerFast): A HuggingFace tokenizer to convert text to token IDs
    """
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for loop_no, filename in enumerate(os.listdir(folder_path), start=1):
            if filename.endswith(".txt"):
                parts = filename.split("_")  # Split filename by '_'

                # Ensure the filename follows the expected format
                if len(parts) < 3:
                    print(f"⚠️ Skipping file (invalid name format): {filename}")
                    continue

                batch_number = parts[1]  # Middle part of filename, used as batch ID

                # Only process files where batch_number == "0"
                if batch_number != "0":
                    continue

                file_path = os.path.join(folder_path, filename)

                # Read file content (each file should contain 256 lines, 4 items per line)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()  # Remove leading/trailing whitespace

                # Tokenize the full file content
                tokens = tokenizer.encode(content)

                # Ensure token length is exactly 1024
                if len(tokens) != 1024:
                    print(f"⚠️ File {filename} has {len(tokens)} tokens after tokenization (expected 1024). Check your data!")
                    continue

                # Write tokenized data as a JSON object (one per line)
                json.dump({"input_ids": tokens}, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")

                # Print progress
                print(f"✅ File {loop_no}: {filename} processed, Tokens: {len(tokens)}")

    print(f"\n🎉 Done! All processed data saved to: {output_jsonl_path} (in JSONL format)")

# === Example usage ===
# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../Tokenizer/Tokenizers/NBtokenizer.json")  # Replace with your tokenizer path

# Define folder and output JSONL path
folder_path = "../../../TOKEN_SLIDING"  # Replace with your input folder
output_jsonl_path = "YuLiaos/yuliao.jsonl"  # Replace with your desired output path

# Run the JSONL conversion process
collect_files_to_jsonl(folder_path, output_jsonl_path, tokenizer)
