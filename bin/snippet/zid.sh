#!/bin/bash

echo "$HOSTNAME"

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
zid_dir="${root_dir}/src/lib/vision/enhance/zid"

cd "${zid_dir}" || exit

python rw_dehazing.py \
  --image "${root_dir}/data/a2i2-haze/train/detection/haze/images" 

cd "${root_dir}" || exit
