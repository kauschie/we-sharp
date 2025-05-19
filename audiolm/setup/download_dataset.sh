#!/bin/bash

# Usage: ./download_dataset.sh links.txt

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <links_file>"
    exit 1
fi

LINKS_FILE="$1"

if [ ! -f "$LINKS_FILE" ]; then
    echo "Error: File '$LINKS_FILE' not found!"
    exit 1
fi

# Create a dataset directory named after the file (without extension)
DATASET_DIR=$(basename "$LINKS_FILE" .txt)
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR" || exit 1

# Download each link and name it using the line number for order
i=1
while IFS= read -r url; do
    printf -v filename "ds1.7z.%03d" "$i"
    echo "Downloading $filename from $url..."
    curl -L "$url" -o "$filename"
    ((i++))
done < "../$LINKS_FILE"

