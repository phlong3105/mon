#!/usr/bin/env bash

set -e

# --- Directories & Files ---
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $(basename "${CURRENT_DIR}") == "scripts" ]; then
    ROOT_DIR=$(dirname "${CURRENT_DIR}")
else
    ROOT_DIR="${CURRENT_DIR}"
fi
PROJECTS_DIR="${ROOT_DIR}/projects"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
ENV_DIR="${SCRIPTS_DIR}/env"

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Creation ---
# Usage: create_relative_alias <source_target> <dest_alias_path>
# Example: create_relative_alias "/path/to/source/file.yaml" "/path/to/dest/"
# Example: create_relative_alias "/path/to/source/file.yaml" "/path/to/dest/symlink.yaml"
create_relative_alias() {
    local source_path="$1"
    local dest_path="$2"

    if [[ -z "$source_path" || -z "$dest_path" ]]; then
        echo "Error: Both source and destination paths are required." >&2
        return 1
    fi

    if [[ ! -e "$source_path" ]]; then
        echo "Error: Source '$source_path' does not exist." >&2
        return 1
    fi

    # Handle directory targets: existing directory OR trailing slash
    local dest_dir dest_name
    if [[ -d "$dest_path" || "$dest_path" == */ ]]; then
        dest_dir="${dest_path%/}"
        dest_name="$(basename "$source_path")"
        dest_path="${dest_dir}/${dest_name}"
    else
        dest_dir="$(dirname "$dest_path")"
        dest_name="$(basename "$dest_path")"
    fi

    mkdir -p "$dest_dir"

    # Compute relative path from destination directory to source
    local rel_target
    rel_target=$(python3 -c "import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$source_path" "$dest_dir")

    # Create the relative symlink
    ln -sfn "$rel_target" "$dest_path"
    echo "==> Created alias: $dest_path -> $rel_target"
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    create_relative_alias
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
