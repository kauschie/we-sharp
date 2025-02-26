import shutil
import os

# Path to the original file
original_file = "./p2-data/micro_test_16khz/beach (2).wav_seg16.wav"

# Output directory (same as original file directory)
output_dir = os.path.dirname(original_file)

# Loop to create 100 copies
for i in range(1, 101):
    new_file_name = f"copy_{i}.wav"
    new_file_path = os.path.join(output_dir, new_file_name)
    
    # Copy the file
    shutil.copy(original_file, new_file_path)

print("✅ 100 copies created successfully in:", output_dir)
