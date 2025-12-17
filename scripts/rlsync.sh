#!/bin/bash

clear
echo "${HOSTNAME}"

# ----- Input -----
option=${1:-"install"}
read -p "Option [install, enable, disable, stop, start]: " -i "$option" -e option

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
root_dir=$current_dir

# ----- Setup -----
install() {
    service_file="${root_dir}/resilio-sync.service"
    target_file="/usr/lib/systemd/user/resilio-sync.service"
    cp "${service_file}" "${target_file}"
}

# ----- Main -----
case "${option}" in
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
        echo "Invalid option: $option"
        exit 1
        ;;
esac

# ----- Done -----
cd "${current_dir}" || exit
exit 0
