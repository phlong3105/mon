#!/bin/bash

# Add/remove a submodule in `mon/projects`.

clear
echo "${HOSTNAME}"

# --- Input ---
GITHUB_USERNAME="longph3105"  # Replace with your GitHub username
REPO_NAME="submodule"  # Replace with the name of the repository you want to add as a submodule
OPTION="add"  # Options: "add" or "remove"

# --- Directory & File ---
CURRENT_FILE=$(readlink -f "${0}")
CURRENT_DIR=$(dirname "${CURRENT_FILE}")
if [ $(basename "${CURRENT_DIR}") == "scripts" ]; then
    ROOT_DIR=$(dirname "${CURRENT_DIR}")
else
    ROOT_DIR="${CURRENT_DIR}"
fi

# --- Main ---
if [ "$OPTION" == "add" ]; then
    git submodule add -f https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git ${ROOT_DIR}/projects/${REPO_NAME}
elif [ "$OPTION" == "remove" ]; then
    git submodule deinit -f ${ROOT_DIR}/projects/${REPO_NAME}
    git rm -f ${ROOT_DIR}/projects/${REPO_NAME}
    rm -rf ${ROOT_DIR}/projects/${REPO_NAME}
fi

# --- Done ---
echo "Submodule processed successfully."
exit 0
