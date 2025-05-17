import os
import json

def extract_unique_numbers_from_files(folder_path):
    """
    遍历文件夹中的所有 .txt 文件，提取所有不重复的数字。

    参数:
        folder_path (str): 文件夹路径。

    返回:
        set: 包含所有不重复数字的集合。
    """
    unique_numbers = set()

    # 遍历文件夹中的所有文件
    i = 0
    for filename in os.listdir(folder_path):
        print(i)
        i += 1
        if filename.endswith(".txt"):  # 只处理 .txt 文件
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                for line in file:
                    # space
                    numbers = line.strip().split(" ")
                    unique_numbers.update(numbers)

    return unique_numbers


# 示例用法
for i in range(8):
    folder_path = rf"C:\Users\HuaiyuZ\PycharmProjects\We_sharp_Jan20\MusicTXT\MusicTxT_{i}"  # 替换为包含 txt 文件的文件夹路径
    unique_numbers = extract_unique_numbers_from_files(folder_path)

    codebook = {number: idx for idx, number in enumerate(unique_numbers)}

    with open(f"CodeBooks\codebook{i}.json", "w") as f:
        json.dump(codebook, f, indent=2)
    print(f"Codebook saved to codebook{i}.json")
