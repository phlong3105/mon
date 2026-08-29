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
check_cuda() {
    if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# --- Lifecycle Management ---
install_xanylabeling() {
    echo -e "\n${YELLOW}Installing X-AnyLabeling Tool...${NC}"
    local xanylabeling_dir="${ROOT_DIR}/tools/xanylabeling"

    if [[ ! -d "$xanylabeling_dir" ]]; then
        mkdir -p "${ROOT_DIR}/tools"
        git clone https://github.com/CVHub520/X-AnyLabeling.git "${xanylabeling_dir}"
    fi

    local PREV_PWD="$(pwd)"
    cd "${xanylabeling_dir}"

    conda create --name xanylabeling python=3.9 -y
    eval "$(conda shell.bash hook)"
    conda activate xanylabeling
    pip install -U pip

    if check_cuda; then
        pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    fi

    case "$OSTYPE" in
        linux*)
            pip install -r requirements-dev.txt
            echo -e "${YELLOW}Installing system dependencies...${NC}"
            sudo apt-get install -y libxcb-xinerama0 || true
            ;;
        darwin*)
            pip install -r requirements-macos-dev.txt
            ;;
    esac

    cd "${PREV_PWD}"
    echo -e "${GREEN}X-AnyLabeling installation complete.${NC}"
}

# --- Service Control ---
start_xanylabeling() {
    local xanylabeling_dir="${ROOT_DIR}/tools/xanylabeling"
    if [[ ! -d "$xanylabeling_dir" ]]; then
        echo -e "${RED}X-AnyLabeling is not installed. Please run the install command first.${NC}"
        exit 1
    fi

    eval "$(conda shell.bash hook)"
    conda activate xanylabeling
    cd "${xanylabeling_dir}" || exit
    python anylabeling/app.py
}

# --- Main Menu ---
manage_xanylabeling() {
    while true; do
        echo -e ""
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}                          X-ANYLABELING TOOL MANAGER                            ${NC}"
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}Host: $(hostname) | OS: $OSTYPE${NC}"
        echo -e "${BLUE}Root: ${ROOT_DIR}${NC}"

        local SUB_OPTIONS=(
            "Install"
            "Start"
            "Exit"
        )
        local DEFAULT_SUB_IDX="1"

        echo -e "\nSelect an OPTION:"
        for i in "${!SUB_OPTIONS[@]}"; do
            printf "  [%d] %s\n" "$i" "${SUB_OPTIONS[i]}"
        done

        read -p "Select Action [default: ${DEFAULT_SUB_IDX}]: " USER_CHOICE
        local SUB_IDX="${USER_CHOICE:-$DEFAULT_SUB_IDX}"
        local ACTION="${SUB_OPTIONS[SUB_IDX]}"
        echo ""

        case "${ACTION}" in
            "Install")
                install_xanylabeling
                read -p "Press Enter to continue..."
                ;;
            "Start")
                start_xanylabeling
                read -p "Press Enter to continue..."
                ;;
            "Exit")
                break
                ;;
            *)
                echo -e "${YELLOW}Invalid option. Please choose a valid index.${NC}"
                ;;
        esac
    done
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    manage_xanylabeling
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
