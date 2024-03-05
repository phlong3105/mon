#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
runml_dir="${bin_dir}/runml"
data_dir="${root_dir}/data"
run_dir="${root_dir}/run"
config_dir="${root_dir}/src/config"
lib_dir="${root_dir}/src/lib"

# Run
model="kind"
data=(
    "dicm" 
    "lime"
    "mef" 
    "npe" 
    "vv"
    "lol-v1"
    "lol-v2-real"
    "lol-v2-syn"
)

cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    # --input-dir "${run_dir}/predict/vision/enhance/llie/${model}/${fullname}/${data}" \
    python -W ignore metric.py \
        --input-dir "${data_dir}/llie/predict/${model}/${data[i]}" \
        --target-dir "${data_dir}/llie/test/${data[i]}/high" \
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