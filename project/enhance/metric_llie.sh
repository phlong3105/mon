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
arch="gcenet_exp_02"
model="gcenet_a1_ablation_20"
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
    input_dir="${data_dir}/${task}/#predict/${arch}/${model}/${data[i]}"
    if ! [ -d "${input_dir}" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}"
    fi

    python -W ignore metric.py \
        --input-dir "${input_dir}" \
        --target-dir "${data_dir}/enhance/${task}/${data[i]}/test/hq" \
        --result-file "${current_dir}" \
        --arch "${arch}" \
        --model "${model}" \
        --device "cuda:0" \
        --metric "psnr" \
        --metric "ssimc" \
        --metric "psnry" \
        --metric "ssim" \
        --metric "lpips" \
        --metric "niqe" \
        --metric "pi" \
        --backend "pyiqa" \
        --use-gt-mean \
        --show-results
done

# Done
cd "${current_dir}" || exit
