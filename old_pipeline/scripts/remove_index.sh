#!/bin/bash

# Check if directory argument is supplied
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

directory="$1"

# Loop through both .m4a and .lrc files in the specified directory
for file in "$directory"/*.{m4a,lrc}; do
  # Skip if no matching files are found
  [ -e "$file" ] || continue
  
  # Remove the numeric prefix from the filename (with test for match)
  newname=$(basename "$file" | sed 's/^[0-9]*\.\s*//')
  
  # Only proceed if the newname is different from the original
  if [ "$newname" != "$(basename "$file")" ]; then
    mv "$file" "$directory/$newname"
    echo "Renamed: $file -> $directory/$newname"
  else
    echo "No change for: $file"
  fi
done
