#!/bin/bash

python detect.py  \
    --weights  weights/yolov7-e6e.pt  \
    --conf 0.25  \
    --img-size 1280  \
    --source /coco/visdrone/VisDrone2019-DET-train/images \
    --save-txt \
    --name yolov7_e6e \
    --project /coco/visdrone/VisDrone2019-DET-train/yolov7_result/
    