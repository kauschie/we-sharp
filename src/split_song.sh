#!/bin/bash

# Directory containing the WAV files
input_dir="."
output_prefix="trimmed-"

# Loop through all files matching the pattern "midi-pp*.wav"
for file in "$input_dir"/midi-pp*.wav; do
    # Extract the filename without the directory path
    filename=$(basename "$file")

    # Construct the output filename
    output_file="$input_dir/$output_prefix$filename"

    # Trim the first 20 seconds of the file
    echo "Processing $file -> $output_file"
    ffmpeg -i "$file" -t 20 -c copy "$output_file"
done

echo "All files processed!"
