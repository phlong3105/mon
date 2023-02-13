#!/bin/bash

echo "$HOSTNAME"

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$(dirname "$current_dir")
zid_dir="${root_dir}/src/lib/zid"

cd "${zid_dir}" || exit

python rw_dehazing.py \
  --image "${root_dir}/data/a2i2-haze/train/detection/haze/images" 

cd "${root_dir}" || exit
