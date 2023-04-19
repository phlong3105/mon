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
createml_dir="${root_dir}/src/mon/createml"

cd "${createml_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python train.py \
      --config "hinet_gt_rain" \
      --root "${root_dir}/run/train" \
      --project "hinet" \
      --name "hinet-gt-rain" \
      --batch-size 4 \
      --image-size 256 \
      --accelerator "auto" \
      --devices 0 \
      --strategy "auto"
  elif [ "$machine" == "vsw-ws01" ]; then
    python train.py \
      --config "hinet_gt_rain" \
      --root "${root_dir}/run/train" \
      --project "hinet" \
      --name "hinet-gt-rain" \
      --batch-size 4 \
      --image-size 512 \
      --accelerator "auto" \
      --devices 0 \
      --strategy "auto"
  elif [ "$machine" == "VSW-WS02" ]; then
    python train.py \
      --config "hinet_rain13k" \
      --root "${root_dir}/run/train" \
      --project "hinet" \
      --name "hinet-rain13k" \
      --weights "NULL" \
      --batch-size 4 \
      --image-size 256 256 \
      --accelerator "auto" \
      --devices 1 \
      --max-epochs "NULL" \
      --max-steps "NULL" \
      --strategy "NULL"
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --config "hinet_rain13k" \
      --root "${root_dir}/run/train" \
      --project "hinet" \
      --name "hinet-rain13k" \
      --weights "NULL" \
      --batch-size 4 \
      --image-size 256 256 \
      --accelerator "auto" \
      --devices 1 \
      --max-epochs "NULL" \
      --max-steps "NULL" \
      --strategy "NULL"
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
fi

cd "${root_dir}" || exist
