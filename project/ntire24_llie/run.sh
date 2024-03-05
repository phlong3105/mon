#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
mon_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
data_dir="${mon_dir}/data"

# Read User Inputs
host=$HOSTNAME
mode=${1:-""}
read -e -i "$mode" -p "Mode [train, predict, metric]: " mode

# Train
if [ "$mode" == "train" ]; then
    cd "${runml_dir}" || exit
    python -W ignore main.py \
        --root "${current_dir}" \
        --task "llie" \
        --mode "train" \
        "$@"

# Predict
elif [ "$mode" == "predict" ]; then
    cd "${runml_dir}" || exit
    python -W ignore main.py \
        --root "${current_dir}" \
        --task "llie" \
        --mode "predict" \
        --data "ntire24_llie" \
        "$@"

# Metric
elif [ "$mode" == "metric" ]; then
    cd "${current_dir}" || exit
fi

# Done
cd "${current_dir}" || exit
