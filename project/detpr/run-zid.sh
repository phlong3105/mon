#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task
# read -e -i "$machine" -p "Machine [pc, server]: " machine

# Initialization
cd "yolov7" || exit

# Install
if [ "$task" == "install" ]; then
  echo -e "\nInstalling"
fi

# Initialization
conda activate mon
cd "zid" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  python rw_dehazing.py \
    --image "../../../data/a2i2-haze/train/detection/haze/images" \
fi

cd ..
