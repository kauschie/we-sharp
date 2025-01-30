import os
import json
from transformers import PreTrainedTokenizerFast

def collect_files_to_json(folder_path, output_json_path, tokenizer, max_no=2200, max_length=1024, split_length=750):
    """
    将文件夹中所有符合条件的文件内容按 tokens 切分并录入 JSON 文件。

    参数:
        folder_path (str): 文件夹路径，文件名格式为 music_<#no>_{i}.txt。
        output_json_path (str): 输出的 JSON 文件路径。
        tokenizer (PreTrainedTokenizerFast): 分词器，用于切分 tokens。
        max_no (int): 最大编号限制，默认为 2200。
        max_length (int): 模型输入上限，默认为 1024。
        split_length (int): 每段的最大 tokens 数，默认为 750。
    """
    collected_data = {}

    for filename in os.listdir(folder_path):
        # 检查文件名是否符合格式 music_<#no>_{i}.txt
        if filename.startswith("music_") and filename.endswith(".txt"):
            try:
                # 提取 <#no> 和 {i}
                parts = filename[6:-4].split('_')
                file_no = int(parts[0])  # 提取 <#no>
                loop_no = int(parts[1])  # 提取 {i}

                # 检查编号是否在范围内
                if file_no <= max_no:
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "r", encoding="utf-8") as file:
                        # 读取文件内容
                        content = file.read().strip()

                        # 分词为 tokens
                        tokens = tokenizer.encode(content)

                        # 打印 token 数量
                        print(f"文件 {filename} 分词后的 token 数量: {len(tokens)}")

                        # 按 tokens 切分
                        chunks = [
                            tokens[i:i + split_length]
                            for i in range(0, len(tokens), split_length)
                        ]

                        # 将每个片段的 token 列表转换为字符串
                        for idx, chunk in enumerate(chunks):
                            chunk_str = " ".join(map(str, chunk))  # 将 token 转换为字符串并用空格分隔
                            print(f"文件 {filename} 的片段 {idx} 长度: {len(chunk)} 内容: {chunk_str[:50]}...")
                            collected_data[f"{file_no}_{loop_no}_{idx}"] = chunk_str
            except (ValueError, IndexError) as e:
                print(f"跳过文件 {filename}，解析失败：{e}")

    # 将收集的数据写入 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(collected_data, json_file, indent=2, ensure_ascii=False, separators=(",", ":"))

    print(f"完成！已将文件内容保存到 {output_json_path}")


# 示例用法
# 加载分词器
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../Tokenizer/Tokenizers/NBtokenizer.json")  # 替换为你的分词器路径

for i in range(8):
    folder_path = rf"C:\Users\HuaiyuZ\PycharmProjects\We_sharp_Jan20\MusicTXT\MusicTxT_{i}"  # 替换为你的文件夹路径
    output_json_path = f"YuLiaos\yuliao_{i}.json"  # 输出的 JSON 文件路径
    collect_files_to_json(folder_path, output_json_path, tokenizer)
