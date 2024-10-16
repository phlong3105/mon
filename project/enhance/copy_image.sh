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
input_dir="${data_dir}/enhance/${task}/#predict"
if ! [ -d "${input_dir}" ]; then
    input_dir="${current_dir}/run/predict"
fi
output_dir="${current_dir}/run/paper"
image_file="lime/3"

# Run
cd "${runml_dir}" || exit

python -W ignore copy_image.py \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --image-file "${image_file}" \
    --imgsz 512

# Done
cd "${current_dir}" || exit
