#!/bin/bash

echo "$HOSTNAME"

option=${1:-"enable"}
read -e -i "$option" -p "Option [install, enable, disable, stop, start]: " option

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$current_dir

#
if [ "${option}" == "install" ]; then
    service_file="${root_dir}/env/resilio-sync.service"
    target_file="/usr/lib/systemd/user/resilio-sync.service"
    cp "${service_file}" "${target_file}"
fi

#
case "${option}" in
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
        ;;
esac

#
echo -e "... Done"
