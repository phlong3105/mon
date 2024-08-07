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

# Run
cd "${runml_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "llie" \
    --mode "predict" \
    --data "1,3,4,9,10,12,6,7,8" \
    --imgsz 504 \
    --verbose \
    "$@"

# Done
cd "${current_dir}" || exit
