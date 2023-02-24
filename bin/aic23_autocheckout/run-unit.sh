#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
unit_dir="${root_dir}/src/lib/unit"

cd "${unit_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  python train.py \
    --config "${unit_dir}/configs/unit_aic23_checkout_synthetic2real.yaml" \
    --output-path "${root_dir}/run/train/unit-aic23-checkout-synthetic2real" \
    --trainer "UNIT"
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
  python test_batch.py \
    --config "${unit_dir}/configs/unit_aic23_checkout_synthetic2real.yaml" \
    --input-folder "${root_dir}/data/a2i2-haze/train/i2i/trainA" \
    --output-folder "${root_dir}/run/predict/unit-aic23-checkout-synthetic2real" \
    --checkpoint "${root_dir}/run/train/unit-aic23-checkout-synthetic2real/outputs/unit_a2i2_hazefree2haze/checkpoints/" \
      --trainer "UNIT"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
fi

cd "${root_dir}" || exit
