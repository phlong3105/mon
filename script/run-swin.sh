#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task
# read -e -i "$machine" -p "Machine [pc, server]: " machine

# Install
if [ "$task" == "install" ]; then
  echo -e "\nInstalling"
fi

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$(dirname "$current_dir")
swin_dir="${root_dir}/src/lib/swin"

cd "${swin_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python tool/train.py \
      configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
      --work-dir "${root_dir}/run/train/cascade_mask_rcnn_swin_tiny_patch4_window7-a2i2" \
      --cfg-options model.pretrained="${root_dir}/zoo/swin/cascade_mask_rcnn_swin_tiny_patch4_window7.pth"
  elif [ "$machine" == "VSW-WS02" ]; then
    python tool/train.py \
      configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
      --work-dir "${root_dir}/run/train/cascade_mask_rcnn_swin_tiny_patch4_window7-a2i2" \
      --cfg-options model.pretrained="${root_dir}/zoo/swin/cascade_mask_rcnn_swin_tiny_patch4_window7.pth"
  elif [ "$machine" == "vsw-ws03" ]; then
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
fi

cd "${root_dir}" || exist
