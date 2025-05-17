import json
import os

# 输入文件路径
input_txt = "../dataPre/token_list.txt"  # 你的TXT文件，每行一个元素
output_folder = "CodeBooks"
output_path = os.path.join(output_folder, "NBcodebook.json")

# 读取TXT文件，并去重 + 排序（如果有需要）
with open(input_txt, "r", encoding="utf-8") as f:
    elements = [line.strip() for line in f if line.strip()]  # 去掉空行

# 生成CodeBook（每个元素映射到唯一索引）
codebook = {str(element): idx for idx, element in enumerate(elements)}

# 创建文件夹并保存JSON
os.makedirs(output_folder, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(codebook, f, indent=2, ensure_ascii=False)

print(f"✅ Codebook saved to {output_path}")
