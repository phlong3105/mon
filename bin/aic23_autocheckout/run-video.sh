#!/bin/bash

machine=$HOSTNAME
echo "$machine"

# Input
dataset=${1:-"aic23-autocheckout"}
subset=${2:-"testA"}
video=${3:-"all"}
preprocess=${4:-"yes"}

read -e -i "$dataset" -p "Dataset [aic23-autocheckout]: " dataset
read -e -i "$subset"  -p "Video [testA, testB]: " subset
if [ "$dataset" == "aic22-autocheckout" ]; then
  read -e -i "$video" -p "Video [${subset}_1, ${subset}_2, ${subset}_3, ${subset}_4, ${subset}_5, all]: " video
else
  read -e -i "$video" -p "Video [${subset}_1, ${subset}_2, ${subset}_3, ${subset}_4, all]: " video
fi
read -e -i "$preprocess" -p "Preprocess [yes, no]: " preprocess

dataset=$(echo $dataset | tr '[:upper:]' '[:lower:]')
subset=$(echo $subset | tr '[:upper:]' '[:lower:]')
video=$(echo $video | tr '[:upper:]' '[:lower:]')
preprocess=$(echo $preprocess | tr '[:upper:]' '[:lower:]')

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
yolov8_dir="${root_dir}/src/lib/yolov8"
lama_dir="${root_dir}/src/lib/lama"
nafnet_dir="${root_dir}/src/lib/nafnet"

video_list=()
if [ "$video" == "all" ]; then
  if [ "$dataset" == "aic22-autocheckout" ]; then
    for var in {1..5}
    do
      video_list+=("${subset}_${var}")
    done
  elif [ "$dataset" == "aic23-autocheckout" ]; then
    for var in {1..4}
    do
      video_list+=("${subset}_${var}")
    done
  fi
else
  video_list=("$video")
fi

# Preprocess
if [ "$preprocess" == "yes" ]; then
  for video in ${video_list[*]}; do
    echo -e "\n$video"

    echo -e "\nConverting video"
    cd "${current_dir}" || exit
    python extract_frame.py \
      --source "${root_dir}/data/${dataset}/${subset}/${video}.mp4" \
      --destination "${root_dir}/data/${dataset}/${subset}/convert"

    echo -e "\nPerforming deblurring"
    cd "${nafnet_dir}" || exit
    python predict.py \
      --source "${root_dir}/data/${dataset}/${subset}/convert/${video}" \
      --destination "${root_dir}/data/${dataset}/${subset}/deblur/${video}" \
      --option "options/test/REDS/NAFNet-width64.yml"

    echo -e "\nGenerating person masks"
    cd "${yolov8_dir}" || exit
    python predict.py \
      --task "segment" \
      --model "${root_dir}/zoo/yolov8/yolov8x-seg-coco.pt" \
      --data "data/coco.yaml" \
      --project "${root_dir}/data/${dataset}/${subset}/person" \
      --name "${video}" \
      --source "${root_dir}/data/${dataset}/${subset}/deblur/${video}" \
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

    # cd "${current_dir}" || exit
    # python postprocess_mask.py \
    #   --mask-dir "${root_dir}/data/${dataset}/${subset}/person/${video}" \
    #   --dilate 5

    echo -e "\nPerforming inpainting"
    cd "${lama_dir}" || exit
    python bin/predict_image_label.py \
      image_dir="${root_dir}/data/${dataset}/${subset}/deblur/${video}" \
      label_dir="${root_dir}/data/${dataset}/${subset}/person/${video}" \
      output_dir="${root_dir}/data/${dataset}/${subset}/inpainting/${video}" \
      model.path="${root_dir}/zoo/lama/big-lama-aic" \
      dataset.kind="video" \
      dataset.dilate=7

    echo -e "\nPerforming background subtraction"
    cd "${current_dir}" || exit
    python gen_background.py \
      --source "${root_dir}/data/${dataset}/${subset}/inpainting/${video}" \
      --destination "${root_dir}/data/${dataset}/${subset}/background/${video}"

    echo -e "\nDetecting tray"
    cd "${current_dir}" || exit
    python detect_tray.py \
      --source "${root_dir}/data/${dataset}/${subset}/background/${video}" \
      --destination "${root_dir}/data/${dataset}/${subset}/tray/${video}"
  done
fi

# Main process
cd "${current_dir}" || exit
python run_camera.py --config "${subset}"

cd "${root_dir}" || exit
