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
check_gui_support() {
    if [[ -n "$DISPLAY" ]] || [[ -n "$WAYLAND_DISPLAY" ]]; then
        return 0
    elif [[ "$OSTYPE" == darwin* ]]; then
        return 0
    else
        return 1
    fi
}

check_cuda() {
    if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

get_env_yaml_path() {
    if check_cuda; then
        echo "${ENV_DIR}/cuda.yaml"
    else
        echo "${ENV_DIR}/cpu.yaml"
    fi
}

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
create_mon_env() {
    echo -e "\n${BLUE}==> Creating 'mon' virtual environment...${NC}"
    local env_yaml
    env_yaml=$(get_env_yaml_path)

    case "$OSTYPE" in
        linux*)
            sudo apt-get install -y gcc g++ || true
            conda env create -f "${env_yaml}"
            add_shell_lines "$HOME/.bashrc" "conda activate mon"
            ;;
        darwin*)
            export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
            export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
            conda env create -f "${env_yaml}"
            add_shell_lines "$HOME/.zshrc" "conda activate mon"
            add_shell_lines "$HOME/.bash_profile" "conda activate mon"
            ;;
    esac

    # Cleanup conflicting OpenCV Qt plugins
    rm -rf "${CONDA_PREFIX}/lib/python3.12/site-packages/cv2/qt/plugins" 2>/dev/null || true
    echo -e "${GREEN}==> 'mon' environment created.${NC}"
}

install_mon_env() {
    echo -e "\n${BLUE}==> Updating 'mon' library dependencies with Poetry...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate mon
    rm -f poetry.lock

    if check_gui_support; then
        poetry install --extras "docs gui"
    else
        poetry install --extras "docs"
    fi

    conda update --all -y
    conda clean --all -y
}

# --- Updating ---
update_conda() {
    echo -e "\n${BLUE}==> Configuring Conda channels and base environment...${NC}"
    conda config --append channels conda-forge || true
    conda config --append channels nvidia || true
    conda config --append channels pytorch || true
    conda update -n base -c defaults conda -y
    pip install --upgrade pip poetry
}

#--- Entry Point ---
