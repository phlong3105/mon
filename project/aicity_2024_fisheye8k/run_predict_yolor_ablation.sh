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
yolor_dir="${mon_dir}/src/mon_extra/vision/detect/yolor"

# Constants
val_cameras=(
    "val/camera1_a_test"
    "val/camera2_a_test"
    "val/camera4_a_e_m_n_test"
    "val/camera7_a_test"
)
conf_thresholds=(
    0.50
    0.50
    0.50
    0.50
)
iou_thresholds=(
    0.50
    0.50
    0.50
    0.50
)
image_sizes=(
    1245
    1088
    1230
    1225
)

# Predict
cd "${yolor_dir}" || exit
save_dir="${current_dir}/run/predict/aicity_2024_fisheye8k/submission"
if [ -d "${save_dir}" ]; then rm -Rf "${save_dir}"; fi
for ((i=0; i < ${#val_cameras[@]}; i++)); do
    source="${data_dir}/aicity/aicity_2024_fisheye8k/${val_cameras[i]}/images"
    python -W ignore my_predict.py \
      --root "${current_dir}" \
      --config "${current_dir}/config/yolor_d6_aicity_2024_fisheye8k_1920.yaml" \
      --weights \
        "${current_dir}/run/train/yolor_d6_ablation_aicity_2024_fisheye8k_bg_igm_ovpl_1920/weights/best_f1.pt" \
      --model "yolor_d6" \
      --data "${source}" \
      --save-dir "${save_dir}" \
      --device "cuda:0" \
      --imgsz 1920 \
      --conf "${conf_thresholds[i]}" \
      --iou "${iou_thresholds[i]}"
done

# Generation submission file
cd "${current_dir}" || exit
python -W ignore gen_submission.py \
    --predict-dir "${save_dir}" \
    --enlarge-ratio 0.0

# Metric
result_file="${current_dir}/run/predict/aicity_2024_fisheye8k/submission/results.json"
gt_file="${data_dir}/aicity/aicity_2024_fisheye8k/val/val.json"
cd "${current_dir}" || exit
python -W ignore metric.py \
    --result-file "${result_file}" \
    --gt-file "${gt_file}"
        
# Done
cd "${current_dir}" || exit
