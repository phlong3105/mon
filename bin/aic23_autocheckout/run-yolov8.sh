#!/bin/bash

echo "$HOSTNAME"

task=$1
machine=$HOSTNAME
read -e -i "$task" -p "Task [install, train, test, predict]: " task

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
      --model "${root_dir}/zoo/yolov8/yolov8l-det-coco.pt" \
      --data "data/aic23-autocheckout-tray.yaml" \
      --project "${root_dir}/run/train/aic23" \
      --name "yolov8l-aic23-autocheckout-tray-640" \
      --epochs 10 \
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
      --data "data/aic23-autocheckout-mix-117.yaml" \
      --project "${root_dir}/run/train/aic23" \
      --name "yolov8x6-aic23-autocheckout-mix-117-1920-02" \
      --epochs 50 \
      --batch 4 \
      --imgsz 1920 \
      --workers 8 \
      --device 0,1 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x-det-coco.pt" \
      --data "data/aic23-autocheckout-testA-117.yaml" \
      --project "${root_dir}/run/train/aic23" \
      --name "yolov8x-aic23-autocheckout-testA-117-1920" \
      --epochs 50 \
      --batch 8 \
      --imgsz 1920 \
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
  if [ "$machine" == "LP-LabDesktop01-Ubuntu" ]; then
    python predict.py \
    	--task "detect" \
    	--model "${root_dir}/run/train/yolov8x-aic23-autocheckout-mix-117-1920/weights/best.pt" \
    	--data "data/aic23-autocheckout-mix-117.yaml" \
    	--project "${root_dir}/run/predict" \
    	--name "testA_3" \
    	--source "${root_dir}/data/aic23-autocheckout/testA/inpainting/testA_3" \
    	--imgsz 1024 \
    	--conf 0.5 \
    	--iou 0.5 \
    	--max-det 3 \
    	--augment \
    	--device 0 \
    	--exist-ok \
    	--save-txt \
    	--overlap-mask \
    	--box
  elif [ "$machine" == "VSW-WS02" ]; then
    python predict.py \
    	--task "detect" \
    	--model "${root_dir}/run/train/yolov8x-aic23-autocheckout-tray-640/weights/best.pt" \
    	--data "data/aic23-autocheckout-tray.yaml" \
    	--project "${root_dir}/run/predict" \
    	--name "yolov8x-aic23-autocheckout-tray-640" \
    	--source "${root_dir}/data/aic23-autocheckout/testA/testA_4.mp4" \
    	--imgsz 640 \
    	--conf 0.9 \
    	--iou 0.5 \
    	--max-det 500 \
    	--augment \
    	--device 0 \
    	--exist-ok \
    	--save \
    	--save-txt \
    	--save-conf \
    	--overlap-mask \
    	--box
  elif [ "$machine" == "vsw-ws03" ]; then
    python predict.py \
    	--task "detect" \
    	--model "${root_dir}/run/train/yolov8l-aic23-autocheckout-mix-117-1920/weights/best.pt" \
    	--data "data/aic23-autocheckout-mix-117.yaml" \
    	--project "${root_dir}/run/predict" \
    	--name "testA_3" \
    	--source "${root_dir}/data/aic23-autocheckout/testA/inpainting/testA_3" \
    	--imgsz 1024 \
    	--conf 0.5 \
    	--iou 0.5 \
    	--max-det 3 \
    	--augment \
    	--device 0 \
    	--exist-ok \
    	--save-txt \
    	--overlap-mask \
    	--box
  fi
fi

cd "${root_dir}" || exit
