#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
archs=(
    #"*"                                 # For all architectures
    ### Specific Architectures
    "zinf"
)
models=(
    #"*"                                 # * for all models
    ### Specific Models
    "zinf"
)
detectors=(
    "deim_dfine_s_coco80"
    #"deim_dfine_s_widerface"
)
datasets=(
    ### High-Level
    #"darkface"
    #"exdark"
    "lolistreetval"
)
bbox_format="yolo"
#exist_ok=$(echo "")
exist_ok=$(echo "--exist-ok")

# ----- Directory & File -----
current_dir=$(pwd)
project_dir=$(dirname "${current_dir}")
root_dir=$(get_root_dir "$(pwd)")
run_dir="${root_dir}/shared/mon/mon_run"

declare -A target_jsons=(
    ["darkface"]="darkface/test/test.json"
    ["exdark"]="exdark/test/test.json"
    ["lolistreetval"]="lolistreet/val/ref.json"
)

declare -A remaps=(
    ["darkface"]=""
    ["exdark"]="exdark/remap_coco802exdark.yaml"
    ["lolistreetval"]=""
)

# ----- Main -----
cd "${run_dir}" || exit

for data in "${datasets[@]}"; do
    # Input
    declare -a input_dirs=()
    for arch in "${archs[@]}"; do
        for model in "${models[@]}"; do
            mapfile -t -O "${#input_dirs[@]}" input_dirs < <(find "${project_dir}/run/predict" -type d -path "*/${arch}/${model}/${data}/pred" 2>/dev/null | sort)
        done
    done
    unique_array "${input_dirs[@]}" input_dirs

    # Target
    target_json="${target_jsons[$data]:-${data}/test/test.json}"
    target_json="${project_dir}/data/${target_json}"
    remap="${remaps[${data}]:-""}"
    [[ "${remap}" != "" ]] && remap="${project_dir}/data/${remap}"

    for input_dir in "${input_dirs[@]}"; do
        data_subdir=$(dirname "${input_dir}")
        model_subdir=$(dirname "${data_subdir}")
        model_name=$(basename "${model_subdir}")
        arch_subdir=$(dirname "${model_subdir}")
        arch_name=$(basename "${arch_subdir}")

        for detector in "${detectors[@]}"; do
            label_dir="${data_subdir}/pred_${detector}/label"
            # Fallback label_dir if not found
            [[ ! -d "${label_dir}" ]] && label_dir="${data_subdir}/pred_${detector}"
            input_json="${data_subdir}/pred_${detector}.json"

            device=$(get_device)

            # Measure COCO
            python -W ignore metric_coco.py \
                --input-dir "${input_dir}" \
                --label-dir "${label_dir}" \
                --input-json "${input_json}" \
                --target-json "${target_json}" \
                --result-file "${project_dir}" \
                --remap "${remap}" \
                --arch "${arch_name}" \
                --model "${model_name}" \
                --data "${data}" \
                --device "${device}" \
                --bbox-format "${bbox_format}" \
                ${exist_ok} \
                "$@"
        done
    done
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
