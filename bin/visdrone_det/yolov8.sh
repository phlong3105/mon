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
yolov8_dir="${root_dir}/src/lib/yolov8"

cd "${yolov8_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "lp-labdesktop-01-ubuntu" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x6-det-coco.pt" \
      --data "data/visdrone.yaml" \
      --project "${root_dir}/run/train/visdrone" \
      --name "yolov8x6-visdrone-1920" \
      --epochs 200 \
      --batch 8 \
      --imgsz 1920 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "vsw-ws02" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x6-det-coco.pt" \
      --data "data/visdrone.yaml" \
      --project "${root_dir}/run/train/visdrone" \
      --name "yolov8x6-visdrone-1920" \
      --epochs 200 \
      --batch 8 \
      --imgsz 1920 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x6-det-coco.pt" \
      --data "data/visdrone.yaml" \
      --project "${root_dir}/run/train/visdrone" \
      --name "yolov8x6-visdrone-1920" \
      --epochs 200 \
      --batch 8 \
      --imgsz 1920 \
      --workers 8 \
      --device 0 \
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
  	--task "detect" \
  	--model "${root_dir}/run/train/visdrone/yolov8x6-visdrone-1920/weights/best.pt" \
  	--data "data/visdrone.yaml" \
  	--project "${root_dir}/run/predict" \
  	--name "syolov8x6-visdrone-1920" \
  	--source "${root_dir}/data/visdrone/test/images" \
  	--imgsz 1920 \
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

cd "${root_dir}" || exit
