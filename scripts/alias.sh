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
    local source_path=""
    local dest_path=""

    echo -e "\033[0;34m==========================================\033[0m"
    echo -e "\033[0;34m        CREATE RELATIVE ALIAS / SYMLINK   \033[0m"
    echo -e "\033[0;34m==========================================\033[0m"

    # 1. Prompt for Source Path (with validation loop)
    while [[ -z "$source_path" ]]; do
        read -p "Enter Source Target Path: " source_path
        # Expand leading tilde (~) if present
        source_path="${source_path/#\~/$HOME}"

        if [[ -z "$source_path" ]]; then
            echo -e "\033[0;31mError: Source path cannot be empty.\033[0m"
        elif [[ ! -e "$source_path" ]]; then
            echo -e "\033[0;31mError: Source '$source_path' does not exist on disk.\033[0m"
            source_path=""
        fi
    done

    # 2. Prompt for Destination Path (with validation loop)
    while [[ -z "$dest_path" ]]; do
        read -p "Enter Destination Shortcut Path (or folder/): " dest_path
        # Expand leading tilde (~) if present
        dest_path="${dest_path/#\~/$HOME}"

        if [[ -z "$dest_path" ]]; then
            echo -e "\033[0;31mError: Destination path cannot be empty.\033[0m"
        fi
    done

    # 3. Determine directory vs. specific target filename
    local dest_dir dest_name
    if [[ -d "$dest_path" || "$dest_path" == */ ]]; then
        dest_dir="${dest_path%/}"
        dest_name="$(basename "$source_path")"
        dest_path="${dest_dir}/${dest_name}"
    else
        dest_dir="$(dirname "$dest_path")"
        dest_name="$(basename "$dest_path")"
    fi

    # Ensure the parent destination directory exists
    mkdir -p "$dest_dir"

    # 4. Calculate relative path from destination directory to source
    local rel_target
    if command -v python3 &>/dev/null; then
        rel_target=$(python3 -c "import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$source_path" "$dest_dir")
    else
        # Pure Bash fallback calculation if python3 is unavailable
        local src_abs dest_dir_abs
        src_abs="$(cd "$(dirname "$source_path")" 2>/dev/null && pwd)/$(basename "$source_path")"
        dest_dir_abs="$(cd "$dest_dir" 2>/dev/null && pwd)"

        IFS='/' read -r -a src_parts <<< "${src_abs#/}"
        IFS='/' read -r -a dest_parts <<< "${dest_dir_abs#/}"

        local common_idx=0
        while [ $common_idx -lt ${#src_parts[@]} ] && [ $common_idx -lt ${#dest_parts[@]} ]; do
            if [ "${src_parts[$common_idx]}" != "${dest_parts[$common_idx]}" ]; then
                break
            fi
            ((common_idx++))
        done

        rel_target=""
        local up_count=$((${#dest_parts[@]} - common_idx))
        for ((i = 0; i < up_count; i++)); do
            rel_target="${rel_target}../"
        done

        for ((i = common_idx; i < ${#src_parts[@]}; i++)); do
            rel_target="${rel_target}${src_parts[$i]}/"
        done
        rel_target="${rel_target%/}"
        rel_target="${rel_target:-.}"
    fi

    # 5. Create the relative symlink (-f forces overwrite, -n handles existing directory symlinks)
    ln -sfn "$rel_target" "$dest_path"

    echo -e "\033[0;32m==> Successfully created relative alias:\033[0m"
    echo -e "    \033[1;33m$dest_path\033[0m -> \033[0;36m$rel_target\033[0m"
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    create_relative_alias
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
