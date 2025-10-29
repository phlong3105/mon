#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
task="lle"
mode="speed"
arch=""
model=""
datasets=(
    "speed10"
    #"speed1k"
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
    --data "${data}" \
    --device "${device}" \
    --imgsz "512"  \
    --resize \
    --benchmark \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
