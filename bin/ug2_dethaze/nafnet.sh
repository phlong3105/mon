#!/bin/bash

echo "$HOSTNAME"

machine=$HOSTNAME
task=$1
read -e -i "$task" -p "Task [install, train, test, predict]: " task

machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
task=$(echo $task | tr '[:upper:]' '[:lower:]')

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
nafnet_dir="${root_dir}/src/lib/nafnet"

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
  cd "${nafnet_dir}" || exit
  python predict.py \
    --source "${root_dir}/data/a2i2-haze/dry-run/2023/images" \
    --destination "${root_dir}/data/a2i2-haze/dry-run/2023/images-denoise" \
    --option "options/test/REDS/NAFNet-width64.yml"
fi

cd "${root_dir}" || exit
