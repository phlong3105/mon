#!/bin/bash

docker run --ipc=host --gpus all \
    --name yolov7 -it  \
    -v /home/vsw/sugar/datasets/:/coco/  \
    -v /home/vsw/sugar/codes/yolov7/:/yolov7  \
    --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3
