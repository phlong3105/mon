#!/bin/bash

subset=$1
video=$2
preprocess=$3
machine=$HOSTNAME
echo "$machine"
read -e -i "$subset" -p "Video [testA, testB]: " subset
read -e -i "$video" -p "Video [${subset}_1, ${subset}_2, ${subset}_3, ${subset}_4, all]: " video
read -e -i "$preprocess" -p "Preprocess [yes, no]: " preprocess

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
yolov8_dir="${root_dir}/src/lib/yolov8"
lama_dir="${root_dir}/src/lib/lama"

if [ "$video" == "all" ]; then
  if [ "$subset" == "testA" ]; then
    video_list=("testA_1" "testA_2" "testA_3" "testA_4")
  else
    video_list=("testB_1" "testB_2" "testB_3" "testB_4")
  fi
else
  video_list=("$video")
fi

# Main loop
for video in ${video_list[*]}; do
  echo -e "\n$video"

  # Preprocess
  if [ "$preprocess" == "yes" ]; then
    echo -e "\nConverting video"
    cd "${current_dir}" || exit
    python convert_video.py \
      --source "${root_dir}/data/aic23-checkout/${subset}/${video}.mp4" \
      --destination "${root_dir}/data/aic23-checkout/${subset}/convert"

    echo -e "\nGenerating person masks"
    cd "${yolov8_dir}" || exit
    python predict.py \
      --task "segment" \
      --model "${root_dir}/zoo/yolov8/yolov8x-seg-coco.pt" \
      --data "data/coco.yaml" \
      --project "${root_dir}/data/aic23-checkout/${subset}/" \
      --name "person" \
      --source "${root_dir}/data/aic23-checkout/${subset}/convert/${video}.mp4" \
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

    echo -e "\nPerforming inpainting"
    cd "${lama_dir}" || exit
    python bin/predict_video.py \
      video_file="${root_dir}/data/aic23-checkout/${subset}/convert/${video}.mp4" \
      label_file="${root_dir}/data/aic23-checkout/${subset}/person/${video}.mp4" \
      output_file="${root_dir}/data/aic23-checkout/${subset}/inpainting/${video}.mp4" \
      model.path="${root_dir}/zoo/lama/big-lama-aic" \
      dataset.kind="video" \
      dataset.dilate=7

    echo -e "\nPerforming background subtraction"
    cd "${current_dir}" || exit
    python gen_background.py \
      --source "${root_dir}/data/aic23-checkout/${subset}/convert/${video}.mp4" \
      --destination "${root_dir}/data/aic23-checkout/${subset}/background/"

    echo -e "\nDetecting tray"
    cd "${current_dir}" || exit
    python detect_tray.py \
      --source "${root_dir}/data/aic23-checkout/${subset}/background/${video}-bg.mp4" \
      --destination "${root_dir}/data/aic23-checkout/${subset}/tray/${video}.mp4"
  fi
done

cd "${root_dir}" || exit
