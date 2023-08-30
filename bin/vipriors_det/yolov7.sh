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

# Install
if [ "$task" == "install" ]; then
  echo -e "\nInstalling YOLOv7"
  sudo nvidia-docker run --name yolov7 -it \
    -v "$root_dir":/mon \
    --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3
  apt update
  apt install -y zip htop screen libgl1-mesa-glx
  pip install seaborn thop pandas
fi

cd "${root_dir}/src/lib/vision/detect/yolov7" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "e600ce73fded" ]; then
    python train_aux.py \
      --weights "" \
      --cfg "cfg/training/yolov7-d6.yaml" \
      --data "data/delftbikes.yaml" \
      --hyp "data/hyp.scratch.custom.yaml" \
      --epochs 200 \
      --batch-size 4 \
      --img-size 1280 \
      --workers 4 \
      --device 0 \
      --sync-bn \
      --exist-ok \
      --project "${root_dir}/run/train/vipriors_det" \
      --name "yolov7-d6-delftbikes-1280" \
      # --resume
  elif [ "$machine" == "cc6e0b2cc23d" ]; then
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train_aux.py \
      --weights "" \
      --cfg "cfg/training/yolov7-e6e.yaml" \
      --data "data/delftbikes.yaml" \
      --hyp "data/hyp.scratch.custom.yaml" \
      --epochs 200 \
      --batch-size 4 \
      --img-size 1920 \
      --workers 4 \
      --device 0,1  \
      --sync-bn \
      --exist-ok \
      --project "${root_dir}/run/train/vipriors_det" \
      --name "yolov7-e6e-delftbikes-1920" \
      # --resume
  elif [ "$machine" == "db2c052f922d" ]; then
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train_aux.py \
      --weights "" \
      --cfg "cfg/training/yolov7-w6.yaml" \
      --data "data/delftbikes.yaml" \
      --hyp "data/hyp.scratch.custom.yaml" \
      --epochs 200 \
      --batch-size 4 \
      --img-size 1920 \
      --workers 4 \
      --device 0,1 \
      --sync-bn \
      --exist-ok \
      --project "${root_dir}/run/train/vipriors_det" \
      --name "yolov7-w6-delftbikes-1920" \
      # --resume
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
  python test.py \
    --weights "run/train/vipriors_det/yolov7-e6e-delftbikes-640/weights/best.pt" \
    --data "data/delftbikes.yaml" \
    --batch-size 8 \
    --img-size 640 \
    --conf-thres 0.00001 \
    --iou-thres 0.5 \
    --device 0 \
    --augment \
    --project "${root_dir}/run/test/delftbikes" \
    --name "yolov7-e6e-delftbikes-640"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  python detect.py \
    --weights \
    "${root_dir}/run/train/vipriors_det/yolov7-d6-delftbikes-1280/weights/best.pt" \
    "${root_dir}/run/train/vipriors_det/yolov7-e6-delftbikes-1920/weights/best.pt" \
    "${root_dir}/run/train/vipriors_det/yolov7-e6e-delftbikes-1920/weights/best.pt" \
    --source "${root_dir}/data/vipriors/delftbikes/test/images" \
    --img-size 2160 \
    --conf-thres 0.0001 \
    --iou-thres 0.5 \
    --device "0" \
    --save-txt \
    --save-conf \
    --nosave \
    --agnostic-nms \
    --augment \
    --project "${root_dir}/run/predict/vipriors_det/prediction" \
    --name "yolov7-ensemble" \
    --exist-ok \
    --no-trace
fi

cd "${root_dir}" || exit
