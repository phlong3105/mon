#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/depthanything_AC_vits.yaml
unlabeled_id_path=partitions/synthetic/full/all.txt
save_path=exp/prior
result_path=exp/prior

mkdir -p $save_path

torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    full_train.py \
    --config=$config  --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --result-path $result_path --port $2 2>&1 | tee $save_path/$now.txt