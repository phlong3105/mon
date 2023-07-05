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
yolov8_dir="${root_dir}/src/lib/yolov8"

cd "${yolov8_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "LP-LabDesktop-01-Ubuntu" ]; then
    python train.py \
      --task "detect" \
      --model "yolov8n.yaml" \
      --data "data/delftbikes.yaml" \
      --project "${root_dir}/run/train/delftbikes" \
      --name "yolov8n-delftbikes-1920" \
      --epochs 500 \
      --batch 8 \
      --imgsz 1920 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      # --pretrained
  elif [ "$machine" == "VSW-WS02" ]; then
    python train.py \
      --task "detect" \
      --model "yolov8x.yaml" \
      --data "data/delftbikes.yaml" \
      --project "${root_dir}/run/train/delftbikes" \
      --name "yolov8x-delftbikes-1920" \
      --epochs 500 \
      --batch 4 \
      --imgsz 1920 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      # --pretrained
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --task "detect" \
      --model "yolov8x6.yaml" \
      --data "data/delftbikes.yaml" \
      --project "${root_dir}/run/train/delftbikes" \
      --name "yolov8x6-delftbikes-1920" \
      --epochs 500 \
      --batch 4 \
      --imgsz 1920 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      # --pretrained
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
  	--task "detect" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x-delftbikes-1920/weights/best.pt" \
  	--data "data/delftbikes.yaml" \
  	--project "${root_dir}/run/predict/delftbikes/" \
  	--name "submission" \
  	--source "${root_dir}/data/vipriors/delftbikes/test/images" \
  	--imgsz 1280 \
  	--conf 0.0001 \
  	--iou 0.5 \
  	--max-det 1000 \
  	--augment \
  	--device 0 \
  	--exist-ok \
  	--save-txt \
  	--save-conf \
  	--overlap-mask \
  	--box
fi

cd "${root_dir}" || exit
