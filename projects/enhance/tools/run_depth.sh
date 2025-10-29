#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
task="depth"
mode="predict"
arch="dav2"
model="dav2_vitb"
datasets=(

)
data=$(printf "%s, " "${datasets[@]}")
data=${data%, }  # Remove trailing ", "

# ----- Directory & File -----
current_dir=$(pwd)
project_dir=$(dirname "${current_dir}")
root_dir=$(get_root_dir "$(pwd)")
run_dir="${root_dir}/shared/mon/mon_run"

# ----- Main -----
cd "${run_dir}" || exit

device=$(get_device)

python -W ignore run.py \
    --root "${project_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --config 0 \
    --data "${data}" \
    --device "${device}" \
    --save-result \
    --save-image \
    --use-fullname \
    --save-nearby \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
