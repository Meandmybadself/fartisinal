#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory does not exist."
    exit 1
fi

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Convert all mp3 files in the input directory to wav files in the output directory
for mp3_file in "$INPUT_DIR"/*.mp3; do
    if [ -f "$mp3_file" ]; then
        base_name=$(basename "$mp3_file" .mp3)
        wav_file="$OUTPUT_DIR/$base_name.wav"
        ffmpeg -i "$mp3_file" "$wav_file"
        echo "Converted: $mp3_file -> $wav_file"
    fi
done

echo "Conversion complete."