#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

if [ $# -lt 3 ]; then
    echo "Usage: $0 <nproc_per_node> <master_port> <dataset>"
    echo "Example: $0 1 29500 kitti"
    echo ""
    echo "Available datasets:"
    echo "  - kitti"
    echo "  - nyu"
    echo "  - sintel"
    echo "  - DIODE"
    echo "  - ETH3D"
    echo "  - robotcar"
    echo "  - nuscene"
    echo "  - foggy"
    echo "  - cloudy"
    echo "  - rainy"
    echo "  - kitti_c_fog"
    echo "  - kitti_c_snow"
    echo "  - kitti_c_dark"
    echo "  - kitti_c_motion"
    echo "  - kitti_c_gaussian"
    echo "  - DA2K"
    echo "  - DA2K_dark"
    echo "  - DA2K_snow"
    echo "  - DA2K_fog"
    echo "  - DA2K_blur"
    exit 1
fi

DATASET=$3

config=configs/synthetic_uncertainty_unsupervised.yaml
torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    evaluate_depth.py \
    --config=$config  \
    --port $2 \
    --dataset $DATASET
    