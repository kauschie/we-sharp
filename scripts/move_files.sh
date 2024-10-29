#!/bin/bash

# Source directory containing the playlists
source_dir=./playlists

# Destination directory where all files will be moved
dest_dir=./ready_to_preprocess

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through all subdirectories in the source directory
for dir in "$source_dir"/*/; do
  # Move all files from the current subdirectory to the destination directory
  mv "$dir"* "$dest_dir"
done

echo "All files have been moved to $dest_dir."
