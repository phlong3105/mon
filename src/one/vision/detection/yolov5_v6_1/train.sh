#!/bin/bash

python train.py \
  --weights '../../../../models_zoo/pretrained/yolov5_v6_1/yolov5x6_v6_1_coco.pt' \
	--cfg yolov5x6.yaml \
	--data aic22retail.yaml \
	--epochs 300 \
	--batch-size 16 \
	--img-size 1536 \
	--device 0,1 \
	--project 'runs/aic22retail' \
	--name 'yolov5x6_v6_1_aic22retail_1536'
