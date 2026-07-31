#!/bin/bash

clear
echo "${HOSTNAME}"

# --- Input ---
OPTION=${1:-"install"}
read -p "Option [install, enable, disable, stop, start]: " -i "$OPTION" -e OPTION

# --- Directory & File ---
CURRENT_FILE=$(readlink -f "${0}")
CURRENT_DIR=$(dirname "${CURRENT_FILE}")
ROOT_DIR=$CURRENT_DIR

# --- Setup ---
install() {
    service_file="${ROOT_DIR}/resilio-sync.service"
    target_file="/usr/lib/systemd/user/resilio-sync.service"
    cp "${service_file}" "${target_file}"
}

# --- Main ---
case "${OPTION}" in
    install)
        install
        ;;
    enable)
        systemctl --user enable resilio-sync
        ;;
    disable)
        systemctl --user disable resilio-sync
        ;;
    stop)
        systemctl --user stop resilio-sync
        ;;
    start)
        systemctl --user start resilio-sync
        ;;
    *)
        echo "Invalid OPTION: $OPTION"
        exit 1
        ;;
esac

# --- Done ---
cd "${CURRENT_DIR}" || exit
exit 0
