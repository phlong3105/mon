#!/usr/bin/env bash

# This script is designed to bootstrap the MON environment on a Linux/macOS system.
# chmod +x bootstrap.sh scripts/*.sh && ./bootstrap.sh

set -e

# --- Directory & File ---
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

# --- Dynamic Module Loader ---
load_module() {
    local module_file="${SCRIPTS_DIR}/$1.sh"
    if [[ -f "${module_file}" ]]; then
        # shellcheck disable=SC1090
        source "${module_file}"
    else
        echo -e "${RED}Error: Module '${1}' not found at ${module_file}${NC}"
        return 1
    fi
}

# --- Main Menu ---
main() {
    while true; do
        clear
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}                            MON ENVIRONMENT BOOTSTRAP                           ${NC}"
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}Host: $(hostname) | OS: $OSTYPE${NC}"
        echo -e "${BLUE}Root: ${ROOT_DIR}${NC}"

        local OPTIONS=(
            "mon (Full Stack)"
            "mon"
            "Submodule"
            "CUDA"
            "Docker"
            "TensorRT"
            "rlsync"
            "X-AnyLabeling"
            "Exit"
        )
        local DEFAULT_OPT_IDX="0"

        echo -e "\nSelect an OPTION:"
        for i in "${!OPTIONS[@]}"; do
            printf "  [%d] %s\n" "$i" "${OPTIONS[i]}"
        done

        read -p "Option [default: ${DEFAULT_OPT_IDX}]: " USER_CHOICE
        local OPT_IDX="${USER_CHOICE:-$DEFAULT_OPT_IDX}"
        local SELECTED="${OPTIONS[OPT_IDX]}"
        echo ""

        case "${SELECTED}" in
            "mon (Full Stack)")
                load_module "system" && quick_setup_system
                load_module "mon" && quick_setup_mon
                load_module "rlsync" && setup_rlsync
                read -p "Press Enter to continue..."
                ;;
            "mon")
                load_module "mon" && manage_mon_env
                ;;
            "Submodule")
                load_module "submodule" && manage_submodule
                ;;
            "CUDA")
                load_module "cuda" && install_nvidia_driver && install_cuda_toolkit
                read -p "Press Enter to continue..."
                ;;
            "Docker")
                load_module "docker" && install_docker
                read -p "Press Enter to continue..."
                ;;
            "TensorRT")
                load_module "tensorrt" && install_tensorrt
                read -p "Press Enter to continue..."
                ;;
            "rlsync")
                load_module "rlsync" && manage_rlsync
                ;;
            "X-AnyLabeling")
                load_module "xanylabeling" && manage_xanylabeling
                read -p "Press Enter to continue..."
                ;;
            "Exit")
                echo -e "${GREEN}Exiting.${NC}"
                break
                ;;
            *)
                echo -e "${RED}Invalid option selected: ${USER_CHOICE}.${NC}"
                sleep 1
                ;;
        esac
    done
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
