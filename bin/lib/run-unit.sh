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
unit_dir="${root_dir}/src/lib/unit"

cd "${unit_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python train.py \
      --config "${unit_dir}/configs/unit_a2i2_hazefree2haze.yaml" \
      --output_path "${root_dir}/run/train/unit-a2i2-hazefree2haze" \
      --trainer "UNIT"
  elif [ "$machine" == "VSW-WS02" ]; then
    python train.py \
      --config "${unit_dir}/configs/unit_a2i2_haze2hazefree.yaml" \
      --output_path "${root_dir}/run/train/unit-a2i2-haze2hazefree" \
      --trainer "UNIT"
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --config "${unit_dir}/configs/unit_a2i2_hazefree2haze.yaml" \
      --output_path "${root_dir}/run/train/unit-a2i2-hazefree2haze" \
      --trainer "UNIT"
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
  python test_batch.py \
      --config "${unit_dir}/configs/unit_a2i2_hazefree2haze.yaml" \
      --input_folder "${root_dir}/data/a2i2-haze/train/i2i/trainA" \
      --output_folder "${root_dir}/run/predict/unit-a2i2-hazefree2haze" \
      --checkpoint "${root_dir}/run/train/unit-a2i2-hazefree2haze/outputs/unit_a2i2_hazefree2haze/checkpoints/" \
      --trainer "UNIT"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
fi

cd "${root_dir}" || exit
