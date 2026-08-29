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

# --- Lifecycle Management ---
install_docker() {
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}Docker is already installed: $(docker --version).${NC}"
    fi

    echo -e "\n${YELLOW}Installing Docker Engine & NVIDIA Container Toolkit...${NC}"
    case "$OSTYPE" in
        linux*)
            sudo apt-get update && sudo apt-get install -y curl
            curl -fsSL https://get.docker.com | sh
            sudo systemctl --now enable docker

            # Setup NVIDIA Container Toolkit if CUDA exists
            if check_cuda; then
                echo -e "${GREEN}Configuring NVIDIA Container Toolkit repository...${NC}"
                local distribution
                distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
                    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                sudo nvidia-ctk runtime configure --runtime=docker
                sudo systemctl restart docker
            fi
            ;;
        darwin*)
            echo -e "${YELLOW}On macOS, please install Docker Desktop via Homebrew: brew install --cask docker${NC}"
            ;;
        *)
            echo -e "${RED}Unsupported platform: $OSTYPE${NC}"
            ;;
    esac
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
