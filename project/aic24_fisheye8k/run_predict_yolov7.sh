#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
mon_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
data_dir="${mon_dir}/data"
yolov7_dir="${mon_dir}/src/mon_lib/vision/detect/yolov7"

# Constants
train_cameras=(
    "train/camera3_a"
    "train/camera3_n"
    "train/camera5_a"
    "train/camera6_a"
    "train/camera8_a"
    "train/camera9_a"
    "train/camera10_a"
    "train/camera11_m"
    "train/camera12_a"
    "train/camera13_a_500"
    "train/camera13_a_779"
    "train/camera14_a"
    "train/camera15_a"
    "train/camera16_a"
    "train/camera17_a"
    "train/camera18_a"
)
val_cameras=(
    "val/camera1_a_test"
    "val/camera2_a_test"
    "val/camera4_a_e_m_n_test"
    "val/camera7_a_test"
)
test_cameras=(
    "test/camera19_a"
    "test/camera20_a"
    "test/camera21_a"
    "test/camera22_a"
    "test/camera23_a"
    "test/camera24_a"
    "test/camera25_a"
    "test/camera26_a"
    "test/camera27_a"
    "test/camera28_a"
    "test/camera29_a_n"
)
conf_thresholds=(
    0.60
    0.60
    0.60
    0.60
    0.60
    0.60
    0.60
    0.60
    0.60
    0.60
    0.60
)
image_sizes=(
    1760
    1760
    1920
    1920
    1920
    2592
    1280
    1280
    1920
    1592
    1944
    1920
)

# Predict
cd "${yolov7_dir}" || exit
save_dir="${current_dir}/run/predict/aic24_fisheye8k/submission"
if [ -d "${save_dir}" ]; then rm -Rf "${save_dir}"; fi
for ((i=0; i < ${#test_cameras[@]}; i++)); do
    source="${data_dir}/aic/aic24_fisheye8k/${test_cameras[i]}/images"
    python -W ignore my_predict.py \
      --root "${current_dir}" \
      --config "${current_dir}/config/yolov7_e6e_aic24_fisheye8k_1920.yaml" \
      --weights \
        "${current_dir}/run/train/yolov7_e6e_aic24_fisheye8k_1920_02/weights/best_f1.pt,
         ${current_dir}/run/train/yolov7_e6e_aic24_fisheye8k_1536_02/weights/best_f1.pt,
         ${current_dir}/run/train/yolov7_e6e_aic24_fisheye8k_1280_02/weights/best_f1.pt"\
      --model "yolov7_e6e" \
      --data "${source}" \
      --save-dir "${save_dir}" \
      --device "cuda:0" \
      --imgsz "${image_sizes[i]}" \
      --conf "${conf_thresholds[i]}" \
      --iou 0.50
done

# Generation submission file
cd "${current_dir}" || exit
python -W ignore gen_submission.py \
    --predict-dir "${save_dir}" \
    --enlarge-ratio 0.0

# Done
cd "${current_dir}" || exit