#!/bin/bash

echo "$HOSTNAME"

machine=$HOSTNAME
task=$1
read -e -i "$task" -p "Task [train, test, predict]: " task

machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
task=$(echo $task | tr '[:upper:]' '[:lower:]')

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
  if [ "$machine" == "lp-labdesktop-01-ubuntu" ]; then
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
  elif [ "$machine" == "lp-labdesktop-02-ubuntu" ]; then
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
  elif [ "$machine" == "vsw-ws02" ]; then
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
