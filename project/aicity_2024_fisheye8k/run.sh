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
    "val_syn/camera1"
    "val_syn/camera2"
    "val_syn/camera7"
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

# Read User Inputs
host=$HOSTNAME
mode=${1:-"train"}
# echo "Mode [train, predict, metric, visualize]: "
# read mode
read -e -i "$mode" -p "Mode [train, predict, metric, visualize]: " mode

# Train
if [ "$mode" == "train" ]; then
    cd "${runml_dir}" || exit
    python -W ignore main.py \
        --root "${current_dir}" \
        --task "detect" \
        --mode "train" \
        "$@"

# Predict
elif [ "$mode" == "predict" ]; then
    save_dir="${current_dir}/run/predict/aicity_2024_fisheye8k/submission"
    declare -a cameras=("${val_cameras[@]}")
    predict=""
    for ((i=0; i < ${#cameras[@]}; i++)); do
        predict+="${data_dir}/aicity/aicity_2024_fisheye8k/${cameras[i]}/images"
        if ((i < ${#cameras[@]} - 1)); then
            predict+=","
        fi
    done
    # Predict
    cd "${runml_dir}" || exit
    if [ -d "${save_dir}" ]; then rm -Rf "${save_dir}"; fi
    python -W ignore main.py \
        --root "${current_dir}" \
        --task "detect" \
        --mode "predict" \
        --data "${predict}" \
        --save-dir "${save_dir}" \
        --imgsz 1280 \
        "$@"
    # Generation submission file
    cd "${current_dir}" || exit
    python -W ignore gen_submission.py \
        --predict-dir "${save_dir}"

# Metric
elif [ "$mode" == "metric" ]; then
    result_file="${current_dir}/run/predict/aicity_2024_fisheye8k/submission/results.json"
    gt_file="${data_dir}/aicity/aicity_2024_fisheye8k/val/val.json"
    cd "${current_dir}" || exit
    python -W ignore metric.py \
        --result-file "${result_file}" \
        --gt-file "${gt_file}"

# Visualize
elif [ "$mode" == "visualize" ]; then
    declare -a cameras=("${val_cameras[@]}")
    for ((i=0; i < ${#cameras[@]}; i++)); do
        python -W ignore visualize_bbox.py \
            --image-dir "${data_dir}/aicity/aicity_2024_fisheye8k/${cameras[i]}/images" \
            --label-dir "${data_dir}/aicity/aicity_2024_fisheye8k/${cameras[i]}/labels" \
            --output-dir "${data_dir}/aicity/aicity_2024_fisheye8k/${cameras[i]}/visualize" \
            --format "yolo" \
            --ext "jpg" \
            --thickness 3 \
            --save
    done
fi

# Done
cd "${current_dir}" || exit
