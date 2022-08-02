#!/bin/bash

python train_export_weight_for_tss.py  \
	--data data/aic22retail.yaml  \
	--cfg yolov5x6.yaml   \
	--weights yolov5x6.pt  \
	--batch-size 2


python src/onedetection/models/yolov5_v4/train.py  \
	--data data/aic22retail.yaml  \
	--cfg yolov5x6.yaml   \
	--weights yolov5x6.pt  \
	--imgsz 1536 \
	--batch-size 8
