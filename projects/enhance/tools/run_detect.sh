#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
task="detect"
mode="predict"
archs=(
    #"*"                                 # For all architectures
    ### Specific Architectures
    "zinf"
)
models=(
    #"*"                                 # * for all models
    ### Specific Models
    "zinf"
)
detector_arch="deim"
detector_model="deim_dfine_s"
datasets=(
    ### High-Level
    #"darkface"
    #"exdark"
    "lolistreetval"
)
#resize=$(echo "")
resize=$(echo "--resize")

# ----- Directory & File -----
current_dir=$(pwd)
project_dir=$(dirname "${current_dir}")
root_dir=$(get_root_dir "$(pwd)")
run_dir="${root_dir}/shared/mon/mon_run"

# ----- Main -----
cd "${run_dir}" || exit

for data in "${datasets[@]}"; do
    # Input
    all_data=""
    for arch in "${archs[@]}"; do
        for model in "${models[@]}"; do
            all_data+=$(find "${project_dir}/run/predict" -type d -path "*/${arch}/${model}/${data}/pred" 2>/dev/null | sort | paste -sd ',' - | sed 's/,$//')
        done
    done

    device=$(get_device)

    python -W ignore run.py \
        --root "${project_dir}" \
        --task "${task}" \
        --mode "${mode}" \
        --arch "${detector_arch}" \
        --model "${detector_model}" \
        --data "${all_data}" \
        --device "${device}" \
        ${resize} \
        --save-result \
        --save-image \
        --save-debug \
        --use-fullname \
        --save-nearby \
        --exist-ok \
        --verbose \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
