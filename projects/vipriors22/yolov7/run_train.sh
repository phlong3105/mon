#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 2  \
    --master_port 9527  \
    train_aux.py --workers 4  \
    --device 0,1  \
    --sync-bn  \
    --batch-size 16  \
    --data data/delftbikes.yaml  \
    --img 1280 1280  \
    --cfg cfg/training/yolov7-e6e.yaml  \
    --weights ''  \
    --name yolov7_e6e  \
    --hyp data/hyp.scratch.p6.yaml

