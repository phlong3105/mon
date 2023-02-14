#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task
# read -e -i "$machine" -p "Machine [pc, server]: " machine

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
imaginaire_dir="${root_dir}/src/lib/imaginaire"

cd "${imaginaire_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  python train.py \
    --config "${imaginaire_dir}/configs/projects/munit/a2i2_hazefree2haze/ampO1.yaml" \
    --logdir "${root_dir}/run/train/munit-a2i2-hazefree2haze"
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
  python test_batch.py \
      --config "${imaginaire_dir}/configs/projects/munit/a2i2_hazefree2haze/ampO1.yaml" \
      --input_folder "${root_dir}/data/a2i2-haze/train/i2i/trainA" \
      --output_folder "${root_dir}/run/predict/munit-a2i2-hazefree2haze" \
      --checkpoint "${root_dir}/run/train/unit-a2i2-hazefree2haze/outputs/unit_a2i2_hazefree2haze/checkpoints/" \
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
fi

cd "${root_dir}" || exit
