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
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
yolov8_dir="${root_dir}/src/lib/yolov8"

cd "${yolov8_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x-det-coco.pt" \
      --data "data/visdrone-a2i2.yaml" \
      --project "${root_dir}/run/train" \
      --name "yolov8x-visdrone-a2i2-640" \
      --epochs 100 \
      --batch 8 \
      --imgsz 640 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "VSW-WS02" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x6-det-coco.pt" \
      --data "data/visdrone-a2i2.yaml" \
      --project "${root_dir}/run/train" \
      --name "yolov8x6-visdrone-a2i2-2160" \
      --epochs 100 \
      --batch 4 \
      --imgsz 2160 \
      --workers 8 \
      --device 0,1 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x-det-coco.pt" \
      --data "data/visdrone-a2i2.yaml" \
      --project "${root_dir}/run/train" \
      --name "yolov8x-visdrone-a2i2-2160" \
      --epochs 200 \
      --batch 4 \
      --imgsz 2160 \
      --workers 8 \
      --device 0,1 \
      --save \
      --exist-ok \
      --pretrained
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  python predict.py \
    --task "segment" \
    --model "${root_dir}/zoo/yolov8/yolov8x-seg-coco.pt" \
    --data "data/coco.yaml" \
    --project "${root_dir}/data/aic23-checkout/run/" \
    --name "testA_1" \
    --source "${root_dir}/data/aic23-checkout/run/testA_1/images/" \
    --imgsz 640 \
    --conf 0.55 \
    --iou 0.5 \
    --max-det 500 \
    --device 0 \
    --exist-ok \
    --save \
    --save-txt \
    --save-conf \
    --overlap-mask \
    --classes 0
fi

cd "${root_dir}" || exist
