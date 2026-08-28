#!/usr/bin/env bash

# reorganize_folders.sh
#
# Reorganizes folder structure by moving folders with a specified suffix (e.g., _dav2)
# into a new directory (e.g., image_<suffix> or ref_<suffix>).
#
# Usage:
#   ./reorganize_folders.sh <parent_dir> <suffix>
#
# Arguments:
#   parent_dir: Path to the parent directory containing image and ref folders.
#   suffix: Suffix of folders to move (e.g., dav2 for folders like 001_dav2).
#
# Example:
#   ./reorganize_folders.sh parent_dir dav2
#
# Before:
#   parent_dir/image/001
#   parent_dir/image/001_dav2
#   parent_dir/image/002
#   parent_dir/image/002_dav2
#   parent_dir/ref/001
#   parent_dir/ref/001_dav2
#
# After (with suffix dav2):
#   parent_dir/image/001
#   parent_dir/image/002
#   parent_dir/image_dav2/001
#   parent_dir/image_dav2/002
#   parent_dir/ref/001
#   parent_dir/ref_dav2/001
#
# Exit Codes:
#   0: Success
#   1: Invalid arguments or directory errors

clear
echo "${HOSTNAME}"

# --- Input ---
# Check if correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <parent_dir> <suffix>"
    exit 1
fi

PARENT_DIR="$1"
SUFFIX="$2"

# Validate parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory $PARENT_DIR does not exist."
    exit 1
fi

# Process both 'image' and 'ref' directories
for MAIN_DIR in "image" "ref"; do
    MAIN_PATH="$PARENT_DIR/$MAIN_DIR"

    # Check if main directory exists
    if [ ! -d "$MAIN_PATH" ]; then
        echo "Directory $MAIN_PATH does not exist. Skipping."
        continue
    fi

    # Create corresponding suffix directory (e.g., image_dav2 or ref_dav2)
    SUFFIX_DIR="$PARENT_DIR/${MAIN_DIR}_${SUFFIX}"
    mkdir -p "$SUFFIX_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create directory $SUFFIX_DIR."
        continue
    fi

    # Find and process folders with the specified suffix
    for FOLDER in "$MAIN_PATH"/*_"$SUFFIX"; do
        if [ -d "$FOLDER" ]; then
            # Extract the base name (e.g., '001' from '001_dav2')
            BASE_NAME=$(basename "$FOLDER" | sed "s/_${SUFFIX}$//")

            # Define target folder path
            TARGET_FOLDER="$SUFFIX_DIR/$BASE_NAME"

            # Move the folder
            mv "$FOLDER" "$TARGET_FOLDER" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "Moved $FOLDER to $TARGET_FOLDER"
            else
                echo "Error moving $FOLDER to $TARGET_FOLDER"
            fi
        fi
    done
done

exit 0
