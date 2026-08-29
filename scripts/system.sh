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
install_conda() {
    if command -v conda &> /dev/null; then
        echo -e "${GREEN}Conda is already installed: $(conda --version).${NC}"
        return 0
    fi

    echo -e "\n${YELLOW}Conda not found. Installing Miniconda...${NC}"
    case "$OSTYPE" in
        linux*)
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
            bash ~/miniconda.sh -b -p $HOME/miniconda
            rm ~/miniconda.sh
            export PATH="$HOME/miniconda/bin:$PATH"
            source ~/.bashrc
            ;;
        darwin*)
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
            bash ~/miniconda.sh -b -p $HOME/miniconda
            rm ~/miniconda.sh
            export PATH="$HOME/miniconda/bin:$PATH"
            source ~/.bash_profile
            ;;
        *)
            echo -e "${YELLOW}Warning: OS $OSTYPE not supported for automated conda setup.${NC}"
            ;;
    esac

    if command -v conda &> /dev/null; then
        echo -e "${GREEN}Conda installation successful.${NC}"
    else
        echo -e "${RED}Conda installation failed. Please install it manually.${NC}"
    fi
}

install_ffmpeg() {
    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}FFmpeg is already installed.${NC}"
        return 0
    fi

    echo -e "\n${YELLOW}FFmpeg not found. Installing FFmpeg...${NC}"
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

    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}FFmpeg installation successful.${NC}"
    else
        echo -e "${RED}FFmpeg installation failed. Please install it manually.${NC}"
    fi
}

install_imagemagick() {
    if command -v convert &> /dev/null; then
        echo -e "${GREEN}ImageMagick is already installed.${NC}"
        return 0
    fi

    echo -e "\n${YELLOW}ImageMagick not found. Installing ImageMagick...${NC}"
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

    if command -v convert &> /dev/null; then
        echo -e "${GREEN}ImageMagick installation successful.${NC}"
    else
        echo -e "${RED}ImageMagick installation failed. Please install it manually.${NC}"
    fi
}

install_turbojpeg() {
    if command -v tjbench &> /dev/null; then
        echo -e "${GREEN}TurboJPEG is already installed.${NC}"
        return 0
    fi

    echo -e "\n${YELLOW}TurboJPEG not found. Installing TurboJPEG...${NC}"
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

    if command -v tjbench &> /dev/null; then
        echo -e "${GREEN}TurboJPEG installation successful.${NC}"
    else
        echo -e "${RED}TurboJPEG installation failed. Please install it manually.${NC}"
    fi
}

# --- Quick Setup ---
quick_setup_system() {
    install_conda
    install_ffmpeg
    install_imagemagick
    install_turbojpeg
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    quick_setup_system
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
