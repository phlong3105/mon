#!/bin/bash

echo "Task [install, train, test, predict]: $1"
echo "Machine [pc, server]: $2"

task=$1
machine=$2
read -e -i "$task" -p "Task [install, train, test, predict]: " task
read -e -i "$machine" -p "Machine [pc, server]: " machine

# Install
if [ "$task" == "install" ]; then
  echo -e "\nInstalling YOLOv4"
  # pip install ultralytics
  if [ "$machine" == "pc" ]; then
    sudo nvidia-docker run --name yolov4 -it \
    -v "/mnt/workspace/mon/data/":/data/ \
    -v "/mnt/workspace/mon/project/detpr/":/detpr \
    --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3
    cd /
    git clone https://github.com/JunnYu/mish-cuda
    cd mish-cuda || exit
    python setup.py build install
  elif [ "$machine" == "server" ]; then
    sudo nvidia-docker run --name yolov4 -it \
    -v "/home/longpham/workspace/mon/data/":/data/ \
    -v "/home/longpham/workspace/mon/project/detpr/":/detpr \
    --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3
    cd /
    git clone https://github.com/JunnYu/mish-cuda
    cd mish-cuda || exit
    python setup.py build install
  fi
fi

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  cd "yolov4" || exit
  if [ "$machine" == "pc" ]; then
    python train.py \
      --weights "weight/yolov4-p7-training.pt" \
      --cfg "yolov4-p7.yaml" \
      --data "visdrone-a2i2-of.yaml" \
      --epochs 100 \
      --batch-size 4 \
      --img 1536 \
      --device 0 \
      --sync-bn \
      --project "../run/train" \
      --name "../run/train/yolov4-p7-visdrone-a2i2-of-1280" \
      # --resume
  elif [ "$machine" == "server" ]; then
    python -m torch.distributed.launch --nproc_per_node 2 train.py \
      --weights "weight/yolov4-p7-training.pt" \
      --cfg "yolov4-p7.yaml" \
      --data "visdrone-a2i2-of.yaml" \
      --epochs 100 \
      --batch-size 4 \
      --img 1536 \
      --device 0,1 \
      --sync-bn \
      --name "../run/train/yolov4-p7-visdrone-a2i2-of-1280" \
      # --resume
  fi
  cd ..
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
  cd "yolov7" || exit
  python test.py \
    --weights "run/train/yolov7-e6e-visdrone-1280/weights/best.pt" \
    --data "data/a2i2.yaml" \
    --batch-size 1 \
    --img-size 1280 \
    --conf-thres 0.00001 \
    --iou-thres 0.5 \
    --device 0 \
    --augment \
    --name yolov7-e6e-visdrone-1280 \
  cd ..
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  cd "yolov7" || exit
  python detect.py \
    --weights "run/train/yolov7-e6e-visdrone-1280/weights/best.pt" \
    --source "../data/ai2i-haze/dry-run/2023/images/" \
    --img-size 1280 \
    --conf-thres 0.00001 \
    --iou-thres 0.5 \
    --agnostic-nms \
    --augment \
    --project "../run/predict" \
    --name yolov7-e6e-visdrone-1280 \
  cd ..
fi
