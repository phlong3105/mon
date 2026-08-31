#!/usr/bin/env bash

# Add/remove a submodule to `mon/projects`.
# Command: chmod +x submodule.sh && ./submodule.sh

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

# --- Defaults ---
DEFAULT_GITHUB_USER="longph3105"  # Replace with your GitHub username

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Utils ---
verify_git_repo() {
    if ! git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree &>/dev/null; then
        echo -e "${RED}Error: ${ROOT_DIR} is not a valid Git repository.${NC}"
        exit 1
    fi
}

# --- Creation ---
add_submodule() {
    echo -e "\n${BLUE}--- Add Git Submodule ---${NC}"

    read -p "GitHub Username/Org [default: ${DEFAULT_GITHUB_USER}]: " INPUT_USER
    local GITHUB_USER="${INPUT_USER:-$DEFAULT_GITHUB_USER}"

    read -p "Repository Name (e.g., my-service): " REPO_NAME
    if [[ -z "${REPO_NAME}" ]]; then
        echo -e "${RED}Error: Repository name cannot be empty.${NC}"
        return 1
    fi

    local REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
    local TARGET_PATH="projects/${REPO_NAME}"

    mkdir -p "${PROJECTS_DIR}"
    cd "${ROOT_DIR}"

    echo -e "${GREEN}Adding submodule from ${REPO_URL} into ${TARGET_PATH}...${NC}"
    if git submodule add -f "${REPO_URL}" "${TARGET_PATH}"; then
        git submodule update --init --recursive "${TARGET_PATH}"
        echo -e "${GREEN}Successfully added submodule: ${TARGET_PATH}${NC}"
    else
        echo -e "${RED}Error: Failed to add submodule.${NC}"
    fi
}

# --- Retrieval ---
list_submodules() {
    echo -e "\n${BLUE}--- Current Submodules Status ---${NC}"
    cd "${ROOT_DIR}"
    if [[ -f ".gitmodules" ]]; then
        git submodule status
    else
        echo -e "${YELLOW}No .gitmodules file present.${NC}"
    fi
}

# --- Updating ---

# --- Deletion ---
remove_submodule() {
    echo -e "\n${BLUE}--- Remove Git Submodule ---${NC}"

    # List active submodules if available
    cd "${ROOT_DIR}"
    if [[ ! -f ".gitmodules" ]] || [[ -z "$(git config --file .gitmodules --get-regexp path)" ]]; then
        echo -e "${YELLOW}No submodules found in this repository.${NC}"
        return 0
    fi

    echo -e "Active submodules:"
    git config --file .gitmodules --get-regexp path | awk '{print "  - " $2}'
    echo ""

    read -p "Enter Submodule Directory Name (under projects/): " REPO_NAME
    if [[ -z "${REPO_NAME}" ]]; then
        echo -e "${RED}Error: Repository name cannot be empty.${NC}"
        return 1
    fi

    local TARGET_PATH="projects/${REPO_NAME}"

    if [[ ! -d "${ROOT_DIR}/${TARGET_PATH}" ]] && ! grep -q "${TARGET_PATH}" .gitmodules 2>/dev/null; then
        echo -e "${RED}Error: Submodule '${TARGET_PATH}' does not exist.${NC}"
        return 1
    fi

    echo -e "${RED}De-initializing submodule ${TARGET_PATH}...${NC}"
    git submodule deinit -f "${TARGET_PATH}" 2>/dev/null || true

    echo -e "${RED}Removing from Git tree and .gitmodules...${NC}"
    git rm -f "${TARGET_PATH}" 2>/dev/null || true

    echo -e "${RED}Purging internal Git cached modules...${NC}"
    rm -rf "${ROOT_DIR}/.git/modules/${TARGET_PATH}"
    rm -rf "${ROOT_DIR}/${TARGET_PATH}"

    echo -e "${GREEN}Submodule ${TARGET_PATH} successfully removed.${NC}"
}

# --- Main Menu ---
manage_submodule() {
    verify_git_repo

    while true; do
        echo -e ""
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}                                SUBMODULE MANAGER                               ${NC}"
        echo -e "${BLUE}================================================================================${NC}"
        echo -e "${BLUE}Host: $(hostname) | OS: $OSTYPE${NC}"
        echo -e "${BLUE}Root: ${ROOT_DIR}${NC}"

        local OPTIONS=(
            "List"
            "Add"
            "Remove"
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
            "List")
                list_submodules
                read -p "Press Enter to continue..."
                ;;
            "Add")
                add_submodule
                ;;
            "Remove")
                remove_submodule
                ;;
            "Exit")
                break
                ;;
            *)
                echo -e "${RED}Invalid option: ${USER_CHOICE}. Choose a valid number.${NC}"
                ;;
        esac
    done
}

# --- Entry Point ---
# Only execute if the script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    manage_submodule
    cd "${CURRENT_DIR}" || exit
    exit 0
fi
