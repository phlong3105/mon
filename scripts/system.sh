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
install_ffmpeg() {
    echo -e "\n${BLUE}==> Installing FFmpeg & Graphics Libraries...${NC}"
    case "$OSTYPE" in
        linux*)
            sudo apt-get update && sudo apt-get install -y \
                ffmpeg '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev \
                libgl1-mesa-glx libxrender-dev libxi-dev libxkbcommon-dev \
                libxkbcommon-x11-dev
            ;;
        darwin*)
            brew install ffmpeg
            ;;
        *)
            echo -e "${YELLOW}Warning: OS $OSTYPE not supported for automated ffmpeg setup.${NC}"
            ;;
    esac
}

install_imagemagick() {
    echo -e "\n${BLUE}==> Installing ImageMagick...${NC}"
    case "$OSTYPE" in
        linux*)
            sudo apt-get install -y imagemagick
            ;;
        darwin*)
            brew install imagemagick
            ;;
        *)
            echo -e "${YELLOW}Warning: OS $OSTYPE not supported for imagemagick.${NC}"
            ;;
    esac
}

install_turbojpeg() {
    echo -e "\n${BLUE}==> Installing TurboJPEG...${NC}"
    case "$OSTYPE" in
        linux*)
            sudo apt-get install -y libturbojpeg
            ;;
        darwin*)
            brew install jpeg-turbo
            ;;
        *)
            echo -e "${YELLOW}Warning: OS $OSTYPE not supported for turbojpeg.${NC}"
            ;;
    esac
}

setup_system() {
    install_ffmpeg
    install_imagemagick
    install_turbojpeg
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_system
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
