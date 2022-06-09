#!/bin/bash

SECONDS=0
# do some work

seqs=(jan mar apr may jun jul aug sep)
gpu_id=0
for seq in ${seqs[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect.py  \
        --pre-cfg "yolov4-p7_chalearnltdmonth_1280"  \
        --months ''${seq}  &
        gpu_id=$(($gpu_id + 1))

    if [ $gpu_id == 2 ]
    then
      gpu_id=0
    fi
done
wait

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
