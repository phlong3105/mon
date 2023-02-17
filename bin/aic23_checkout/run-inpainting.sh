#!/bin/bash

echo "$HOSTNAME"

video=$1
convert=$2
segmentation=$3
inpainting=$4
machine=$HOSTNAME
read -e -i "$video" -p "Video [testA_1, testA_2, testA_3, testA_4, all]: " video
read -e -i "$convert" -p "Convert video [yes, no]: " convert
read -e -i "$segmentation" -p "Segmentation method [yolov8, no]: " segmentation
read -e -i "$inpainting" -p "Inpainting method [lama, e2fgvi, no]: " inpainting

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
yolov8_dir="${root_dir}/src/lib/yolov8"
lama_dir="${root_dir}/src/lib/lama"
e2fgvi_dir="${root_dir}/src/lib/e2fgvi"

if [ "$video" == "all" ]; then
  video_list=("testA_1" "testA_2" "testA_3" "testA_4")
else
  video_list=("$video")
fi

for video in ${video_list[*]}; do
  echo "$video"

  if [ "$convert" == "yes" ]; then
    echo -e "\nConverting video"
    cd "${current_dir}" || exit
    python convert_video.py \
      --source "${root_dir}/data/aic23-checkout/testA/${video}.mp4" \
      --output-dir "${root_dir}/data/aic23-checkout/testA/convert"
  fi

  if [ "$segmentation" == "yolov8" ]; then
    echo -e "\nGenerating person masks"
    cd "${yolov8_dir}" || exit
    python predict.py \
      --task "segment" \
      --model "${root_dir}/zoo/yolov8/yolov8x-seg-coco.pt" \
      --data "data/coco.yaml" \
      --project "${root_dir}/data/aic23-checkout/testA/" \
      --name "person" \
      --source "${root_dir}/data/aic23-checkout/testA/convert/${video}.mp4" \
      --imgsz 640 \
      --conf 0.1 \
      --iou 0.1 \
      --max-det 300 \
      --device 0 \
      --stream \
      --exist-ok \
      --save \
      --save-mask \
      --retina-masks \
      --classes 0
  fi
  # cd "${current_dir}" || exit
  # python gen_human_mask.py \
  #   --image-dir "${root_dir}/data/aic23-checkout/testA/${video}/images/" \
  #   --label-dir "${root_dir}/data/aic23-checkout/testA/${video}/person/" \
  #   --segment-format "yolo" \
  #   --dilate 7 \
  #   --thickness -1 \
  #   --extension "jpg" \
  #   --save
  if [ "$inpainting" == "lama" ]; then
    echo -e "\nPerforming inpainting"
    cd "${lama_dir}" || exit
    # python bin/predict.py \
    #   indir="${root_dir}/data/aic23-checkout/testA/${video}/person-masks/" \
    #   outdir="${root_dir}/data/aic23-checkout/testA/${video}/inpainting/" \
    #   model.path="${root_dir}/zoo/lama/big-lama-aic" \
    #   dataset.dilate=7
    python bin/predict_video.py \
      video_file="${root_dir}/data/aic23-checkout/testA/convert/${video}.mp4" \
      label_file="${root_dir}/data/aic23-checkout/testA/person/${video}.mp4" \
      output_file="${root_dir}/data/aic23-checkout/testA/inpainting/${video}.mp4" \
      model.path="${root_dir}/zoo/lama/big-lama-aic" \
      dataset.kind="video" \
      dataset.dilate=7
  else
    echo -e "\nPerforming inpainting"
    cd "${e2fgvi_dir}" || exit
    python test.py \
      --video "${root_dir}/data/aic23-checkout/testA/convert/${video}.mp4" \
      --ckpt "${root_dir}/zoo/e2fgvi/e2fgvi-cvpr22.pth" \
      --mask "${root_dir}/data/aic23-checkout/testA/person/${video}.mp4" \
      --output "${root_dir}/data/aic23-checkout/testA/e2fgvi/${video}.mp4" \
      --model "e2fgvi"
  fi
done

cd "${root_dir}" || exist
