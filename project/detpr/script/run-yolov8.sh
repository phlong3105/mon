#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task
# read -e -i "$machine" -p "Machine [pc, server]: " machine

# Install
if [ "$task" == "install" ]; then
  echo -e "\nInstalling YOLOv8"
fi

# Initialization
conda activate mon
cd "src/yolov8" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python train.py \
      --task "detect" \
      --mode "train" \
      --model "weight/yolov8x-det-coco.pt" \
      --data "data/visdrone-a2i2.yaml" \
      --epochs 100 \
      --batch 8 \
      --imgsz 640 \
      --device 0 \
      --workers 8 \
      --save \
      --exist-ok \
      --pretrained \
      --project "../../run/train" \
      --name "yolov8x-visdrone-a2i2-640" \
      # --resume
  elif [ "$machine" == "VSW-WS02" ]; then
    python train.py \
      --task "detect" \
      --mode "train" \
      --model "weight/yolov8x-det-coco.pt" \
      --data "data/visdrone-a2i2-of.yaml" \
      --epochs 100 \
      --batch 4 \
      --imgsz 2160 \
      --device 0,1 \
      --workers 8 \
      --save \
      --exist-ok \
      --pretrained \
      --project "../../run/train" \
      --name "yolov8x-visdrone-a2i2-of-2160" \
      # --resume
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --task "detect" \
      --mode "train" \
      --model "weight/yolov8x-det-coco.pt" \
      --data "data/visdrone-a2i2.yaml" \
      --epochs 200 \
      --batch 4 \
      --imgsz 2160 \
      --device 0,1 \
      --workers 8 \
      --save \
      --exist-ok \
      --pretrained \
      --project "../../run/train" \
      --name "yolov8x-visdrone-a2i2-2160" \
      # --resume
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

cd ..
