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

# --- Installation ---
check_cuda() {
    if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

install_tensorrt() {
    echo -e "\n${BLUE}==> Installing TensorRT & Building trtexec...${NC}"
    if ! check_cuda; then
        echo -e "${RED}Error: CUDA is not detected. Please install CUDA first.${NC}"
        return 1
    fi

    sudo apt-get update
    sudo apt-get install -y tensorrt onnx-graphsurgeon
    sudo apt autoremove -y

    if [[ -d "/usr/src/tensorrt/samples/trtexec" ]]; then
        local PREV_PWD="$(pwd)"
        cd /usr/src/tensorrt/samples/trtexec
        sudo make CUDA_INSTALL_DIR=/usr/local/cuda/bin TRT_LIB_DIR=/usr/src/tensorrt/bin
        sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/
        cd "${PREV_PWD}"
        echo -e "${GREEN}trtexec successfully built and installed to /usr/local/bin/trtexec${NC}"
    else
        echo -e "${YELLOW}Notice: TensorRT sample directory not found; skipping manual trtexec compile.${NC}"
    fi
}

#--- Entry Point ---
