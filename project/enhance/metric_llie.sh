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

# Input
task="llie"
model="rapidlight_v01"
data=(
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    "lol_v1"
    "lol_v2_real"
    "lol_v2_synthetic"
)

# Run
cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    input_dir="${data_dir}/${task}/#predict/${model}/${data[i]}"
    if ! [ -d "${input_dir}" ]; then
        input_dir="${current_dir}/run/predict/${model}/${data[i]}"
    fi

    python -W ignore metric.py \
        --input-dir "${input_dir}" \
        --target-dir "${data_dir}/${task}/${data[i]}/test/hq" \
        --result-file "${current_dir}" \
        --name "${model}" \
        --metric "psnr" \
        --metric "ssim" \
        --metric "lpips" \
        --metric "niqe" \
        --metric "pi" \
        --test-y-channel \
        --backend "pyiqa" \
        --show-results
done

# Done
cd "${current_dir}" || exit
