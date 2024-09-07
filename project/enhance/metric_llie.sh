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
arch="zeroig"
model="zeroig_re"
data=(
    # "dicm"
    # "lime"
    # "mef"
    # "nightcity" 
    # "npe"
    # "vv"
    "lol_v1"
    # "lol_v2_real"
    # "lol_v2_synthetic"
)

# Run
cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    input_dir="${data_dir}/${task}/#predict/${arch}/${model}/${data[i]}"
    if ! [ -d "${input_dir}" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}"
    fi
    target_dir="${data_dir}/enhance/${task}/${data[i]}/test/hq"
    if ! [ -d "${target_dir}" ]; then
        target_dir="${data_dir}/enhance/${task}/${data[i]}/val/hq"
    fi

    python -W ignore metric.py \
        --input-dir "${input_dir}" \
        --target-dir "${target_dir}" \
        --result-file "${current_dir}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${data[i]}" \
        --device "cuda:0" \
        --metric "psnr" \
        --metric "ssimc" \
        --metric "psnry" \
        --metric "ssim" \
        --metric "lpips" \
        --metric "niqe" \
        --metric "pi" \
        --backend "pyiqa" \
        --show-results
        # --use-gt-mean \
done

# Done
cd "${current_dir}" || exit
