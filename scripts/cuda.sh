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

# --- Utils ---
add_shell_lines() {
    local target_file="$1"
    shift
    local lines=("$@")

    echo -e "${BLUE}Checking configuration in ${target_file}...${NC}"
    touch "${target_file}"
    for line in "${lines[@]}"; do
        if grep -Fxq "$line" "$target_file"; then
            echo -e "  Already configured: ${line}"
        else
            echo -e "  ${GREEN}+ Adding line:${NC} ${line}"
            echo "$line" >> "$target_file"
        fi
    done
}

# --- Installation ---
install_nvidia_driver() {
    echo -e "\n${BLUE}==> Installing NVIDIA Drivers (Linux)...${NC}"
    if [[ "$OSTYPE" != linux* ]]; then
        echo -e "${YELLOW}NVIDIA Driver installation is only supported on Linux hosts.${NC}"
        return 0
    fi
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y gcc g++ ubuntu-drivers-common
    sudo ubuntu-drivers devices
    sudo apt install -y nvidia-driver-570
}

install_cuda_toolkit() {
    echo -e "\n${BLUE}==> Installing CUDA Toolkit 12.6...${NC}"
    if [[ "$OSTYPE" != linux* ]]; then
        echo -e "${YELLOW}CUDA Toolkit installation is only supported on Linux hosts.${NC}"
        return 0
    fi

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6

    local bashrc_lines=(
        'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
        'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'
    )
    add_shell_lines "$HOME/.bashrc" "${bashrc_lines[@]}"
    echo -e "${GREEN}CUDA installation completed. Remember to run: source ~/.bashrc${NC}"
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
