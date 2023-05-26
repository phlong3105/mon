#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [train, test, predict]: " task

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
nafnet_dir="${root_dir}/src/lib/nafnet"

cd "${nafnet_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPerforming deblurring"
  python predict.py \
    --source "${root_dir}/data/aic23-autocheckout/train/patches" \
    --destination "${root_dir}/data/aic23-autocheckout/train/patches-deblur" \
    --option "options/test/REDS/NAFNet-width64.yml" \
    --extension "png"
fi

cd "${root_dir}" || exit
