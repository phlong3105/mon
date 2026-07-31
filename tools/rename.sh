#!/bin/bash

# Rename files and directories in a specified directory to a snake_case format
# (lowercase, replace spaces and hyphens with underscores, and reduce multiple
# underscores to a single underscore).

clear
echo "${HOSTNAME}"

# --- Input ---
DIRECTORY="/home/longpham/10_workspace/11_code/mon"

# --- Functions ---
normalize_name() {
    # Function to normalize names (lowercase, replace ' ' and '-' with '_', and reduce '__' to '_')
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr '-' '_' | sed 's/__\+/_/g'
}

# --- Main ---
# Validate the DIRECTORY
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: '$DIRECTORY' is not a valid DIRECTORY."
    exit 1
fi

# Process both files and directories in one pass
find "$DIRECTORY" -depth | while IFS= read -r path; do
    # Skip if it's the target DIRECTORY itself
    if [ "$path" = "$DIRECTORY" ]; then
        continue
    fi

    # Extract DIRECTORY and base name
    dir=$(dirname "$path")
    old_name=$(basename "$path")

    # Generate new normalized name
    new_name=$(normalize_name "$old_name")

    # Rename if the name has changed
    if [ "$old_name" != "$new_name" ]; then
        mv -v "$dir/$old_name" "$dir/$new_name"
    fi
done

# --- Done ---
echo "Renaming process completed."
exit 0
