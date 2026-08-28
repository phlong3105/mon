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
check_linux() {
    if [ "$(uname -s)" != "Linux" ]; then
        echo -e "${RED}Error: This script is intended for Linux environments only.${NC}"
        exit 1
    fi
}

detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    else
        echo -e "${RED}Error: Cannot detect Linux distribution (/etc/os-release missing).${NC}"
        exit 1
    fi
}

# --- Installation ---
install_rlsync() {
    check_linux
    detect_distro

    if command -v rslsync &> /dev/null; then
        echo -e "${GREEN}==> Resilio Sync is already installed ($(rslsync --help | head -n 1)).${NC}"
        return 0
    fi

    echo -e "${BLUE}==> Installing Resilio Sync on ${DISTRO}...${NC}"

    case "$DISTRO" in
        ubuntu|debian|pop|linuxmint)
            echo -e "${GREEN}Configuring official Resilio APT repository...${NC}"
            sudo apt-get update
            sudo apt-get install -y curl gnupg lsb-release
            # Register GPG key and APT repository
            curl -fsSL https://linux-packages.resilio.com/resilio-sync/key.asc | sudo gpg --dearmor -o /usr/share/keyrings/resilio-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/resilio-archive-keyring.gpg] http://linux-packages.resilio.com/resilio-sync/deb resilio-sync non-free" | sudo tee /etc/apt/sources.list.d/resilio-sync.list
            sudo apt-get update
            sudo apt-get install -y resilio-sync
            ;;
        arch|manjaro)
            echo -e "${GREEN}Installing Resilio Sync via AUR helper (yay/paru)...${NC}"
            if command -v yay &> /dev/null; then
                yay -Sy --noconfirm rslsync
            elif command -v paru &> /dev/null; then
                paru -Sy --noconfirm rslsync
            else
                echo -e "${RED}Error: Neither 'yay' nor 'paru' was found. Please install an AUR helper or install 'rslsync' manually.${NC}"
                return 1
            fi
            ;;
        fedora|rhel|centos)
            echo -e "${GREEN}Configuring official Resilio RPM repository...${NC}"
            sudo rpm --import https://linux-packages.resilio.com/resilio-sync/key.asc
            zypper ar --gpgcheck-allow-unsigned-repo -f https://linux-packages.resilio.com/resilio-sync/rpm/\$basearch resilio-sync
            printf "[resilio-sync]\nname=Resilio Sync\nbaseurl=https://linux-packages.resilio.com/resilio-sync/rpm/\$basearch\nenabled=1\ngpgcheck=1\n" | sudo tee /etc/yum.repos.d/resilio-sync.repo
            sudo yum install resilio-sync
            ;;
        *)
            echo -e "${RED}Unsupported Linux distribution: $DISTRO${NC}"
            echo -e "${YELLOW}Please visit https://www.resilio.com/individuals-sync/ to download standalone binaries.${NC}"
            return 1
            ;;
    esac

    echo -e "${GREEN}==> Resilio Sync installed successfully!${NC}"
}

setup_rlsync() {
    echo -e "\n${BLUE}==> Configuring Resilio Sync metadata & IgnoreList...${NC}"
    local rsync_dir="${ROOT_DIR}/.sync"
    mkdir -p "${rsync_dir}"

    if [[ -f "${ENV_DIR}/IgnoreList" ]]; then
        cp "${ENV_DIR}/IgnoreList" "${rsync_dir}/IgnoreList"
        echo -e "${GREEN}IgnoreList deployed to ${rsync_dir}/IgnoreList${NC}"
    else
        echo -e "${YELLOW}Warning: ${ENV_DIR}/IgnoreList not found.${NC}"
    fi
}

# --- Management ---
enable_user_mode() {
    echo -e "${BLUE}==> Enabling Resilio Sync as current user ($USER)...${NC}"

    # Enable Resilio Sync under the current user's systemd session
    systemctl --user enable --now resilio-sync || {
        echo -e "${YELLOW}User systemd service unavailable. Enabling system-wide service...${NC}"
        sudo systemctl enable --now resilio-sync
    }

    # Retrieve local IP
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    LOCAL_IP="${LOCAL_IP:-localhost}"

    echo -e "\n${GREEN}==========================================${NC}"
    echo -e "${GREEN}      RESILIO SYNC SERVICE READY!         ${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo -e "Access the Web GUI at: ${BLUE}http://${LOCAL_IP}:8888${NC} or ${BLUE}http://localhost:8888${NC}"
}

disable_user_mode() {
    echo -e "${BLUE}==> Disabling Resilio Sync for current user ($USER)...${NC}"
    systemctl --user disable resilio-sync || {
        echo -e "${YELLOW}User systemd service unavailable. Disabling system-wide service...${NC}"
        sudo systemctl disable resilio-sync
    }
}

start_rlsync() {
    if systemctl --user is-enabled resilio-sync &>/dev/null; then
        systemctl --user start resilio-sync
    else
        sudo systemctl start resilio-sync
    fi
    echo -e "${GREEN}Resilio Sync started.${NC}"
}

stop_rlsync() {
    if systemctl --user is-active resilio-sync &>/dev/null; then
        systemctl --user stop resilio-sync
    else
        sudo systemctl stop resilio-sync
    fi
    echo -e "${RED}Resilio Sync stopped.${NC}"
}

show_rlsync_status() {
    if systemctl --user is-active resilio-sync &>/dev/null; then
        systemctl --user status resilio-sync
    else
        sudo systemctl status resilio-sync
    fi
}

# --- Main Menu ---
manage_rlsync() {
    while true; do
        echo -e ""
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}                           RESILIO SYNC SERVICE MANAGER                         ${NC}"
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}Host: $(hostname) | OS: $OSTYPE${NC}"
        echo -e "${BLUE}Root: ${ROOT_DIR}${NC}"

        local SUB_OPTIONS=(
            "Start RLSync"
            "Stop RLSync"
            "Install RLSync"
            "Setup RLSync"
            "Disable RLSync"
            "Show Status"
            "Back to Main Menu"
        )
        local DEFAULT_SUB_IDX="0"

        echo -e "\nSelect an OPTION:"
        for i in "${!SUB_OPTIONS[@]}"; do
            printf "  [%d] %s\n" "$i" "${SUB_OPTIONS[i]}"
        done

        read -p "Select Action [default: ${DEFAULT_SUB_IDX}]: " USER_CHOICE
        local SUB_IDX="${USER_CHOICE:-$DEFAULT_SUB_IDX}"
        local ACTION="${SUB_OPTIONS[SUB_IDX]}"
        echo ""

        case "${ACTION}" in
            "Start RLSync")
                start_rlsync
                ;;
            "Stop RLSync")
                stop_rlsync
                ;;
            "Install RLSync")
                install_rlsync
                setup_rlsync
                enable_user_mode
                start_rlsync
                ;;
            "Setup RLSync")
                setup_rlsync
                enable_user_mode
                ;;
            "Disable RLSync")
                disable_user_mode
                ;;
            "Show Status")
                show_rlsync_status
                ;;
            "Back to Main Menu")
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
    manage_rlsync
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
