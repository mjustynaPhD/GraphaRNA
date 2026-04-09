#!/bin/bash

# Define the file pattern to search for
FILE_PATTERN="*.gz"

# Check if any matching files exist
if ! ls $FILE_PATTERN 1> /dev/null 2>&1; then
    echo "No files matching '$FILE_PATTERN' found in the current directory."
    exit 0
fi

# Inform the user what is about to happen
echo "Starting decompression for all '$FILE_PATTERN' files..."

# Loop through all files matching the pattern
for file in $FILE_PATTERN; do
    echo "Decompressing: $file"
    # Run the gunzip command. gunzip will replace the *.gz file with the decompressed file.
    gunzip "$file"
    # Check the exit status of the gunzip command
    if [ $? -eq 0 ]; then
        echo "Successfully decompressed."
    else
        echo "Error decompressing $file."
    fi
done

echo "---"
echo "Decompression process finished."