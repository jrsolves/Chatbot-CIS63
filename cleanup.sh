#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Define the path to the static directory
STATIC_DIR="$SCRIPT_DIR/static"

# Find and delete .mp4 and .mp3 files older than 1 hour (60 minutes)
find "$STATIC_DIR" -type f \( -name "*.mp4" -o -name "*.mp3" \) -mmin +60 -exec rm {} \;
