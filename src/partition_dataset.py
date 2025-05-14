import os
import shutil

# paths for testing
base = "/mnt/c/Users/mkaus/Downloads/The_Magic_of_MIDI/"
source_dir = os.path.join(base, "done/MIDI")
target_dirs = ["set_1", "set_2", "set_3", "set_4", "set_5"]
files = os.listdir(source_dir)
chunks = len(target_dirs)

for i, file in enumerate(files):
    # Get the full path of the current file
    file_path = os.path.join(source_dir, file)

    # Check if it's a file and has a .mid or .MID extension
    if os.path.isfile(file_path) and file.lower().endswith(".mid"):
        # Move the file to the appropriate target directory
        shutil.move(file_path, os.path.join(target_dirs[i % chunks], file))