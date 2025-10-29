#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
archs=(
    #"*"                                 # For all architectures
    ### Specific Architectures
    "gcenet"
)
models=(
    "*"                                 # * for all models
    ### Specific Models
    # "gcenet"
)
datasets=(
    ### Unpaired
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    ### LOLs
    "lolv1"
    "lolv2real"
    "lolv2syn"
    ### LSRW
    "lsrw"
    ### SICE
    "sice"
    ### FiveK
    #"fiveka"
    #"fivekb"
    #"fivekc"
    #"fivekd"
    "fiveke"
    ### UHD
    "uhdll"
    ### High-Level
    "darkface"
    "exdark"
    "lolistreettest"
    "lolistreetval"
    "ydld"
)
imgsz=512
resize=$(echo "")
#resize=$(echo "--resize")
use_gt_mean=$(echo "")
# use_gt_mean=$(echo "--use-gt-mean")
metrics=(
    "psnr"
    "ssimc"
    "lpips"
    "ilniqe"
    "niqe"
    "pi"
)

# ----- Directory & File -----
current_dir=$(pwd)
project_dir=$(dirname "${current_dir}")
root_dir=$(get_root_dir "$(pwd)")
run_dir="${root_dir}/shared/mon/mon_run"

declare -A input_subdirs=(
    ["fiveka"]="fivek/pred"
    ["fivekb"]="fivek/pred"
    ["fivekc"]="fivek/pred"
    ["fivekd"]="fivek/pred"
    ["fiveke"]="fivek/pred"
)
declare -A target_subdirs=(
    ["lolv2real"]="lolv2/real/test/ref"
    ["lolv2syn"]="lolv2/syn/test/ref"
    ["sice"]="sice/sice/test/ref"
    ["fiveka"]="fivek/test/ref_a"
    ["fivekb"]="fivek/test/ref_b"
    ["fivekc"]="fivek/test/ref_c"
    ["fivekd"]="fivek/test/ref_d"
    ["fiveke"]="fivek/test/ref_e"
    ["lolistreetval"]="lolistreet/val/ref"
)

# ----- Main -----
cd "${run_dir}" || exit

# Parse metrics arguments
metric_args=()
for metric in "${metrics[@]}"; do
    metric_args+=("--metric" "${metric}")
done

for data in "${datasets[@]}"; do
    # Input
    declare -a input_dirs=()
    input_subdir="${input_subdirs[${data}]:-${data}/pred}"
    for arch in "${archs[@]}"; do
        for model in "${models[@]}"; do
            mapfile -t -O "${#input_dirs[@]}" input_dirs < <(find "${project_dir}/run/predict" -type d -path "*/${arch}/${model}/${input_subdir}" 2>/dev/null | sort)
        done
    done
    # unique_array "${input_dirs[@]}" input_dirs

    # Target
    target_subdir="${target_subdirs[$data]:-${data}/test/ref}"
    target_dir="${project_dir}/data/${target_subdir}"
    # Fallback target_dir if not found
    [[ ! -d "${target_dir}" ]] && target_dir="${root_dir}/data/enhance/${target_subdir}"

    # Determine IQA type
    if [[ ${use_gt_mean} == "--use-gt-mean" ]]; then
        use_gt_mean=$([[ -d "${target_dir}" ]] && echo "--use-gt-mean" || echo "")
    fi
    if [[ ${resize} == "--resize" ]]; then
        # Using resize with NR-IQA is not recommended.
        resize=$([[ -d "${target_dir}" ]] && echo "--resize" || echo "")
    fi

    for input_dir in "${input_dirs[@]}"; do
        data_subdir=$(dirname "${input_dir}")
        model_subdir=$(dirname "${data_subdir}")
        model_name=$(basename "${model_subdir}")
        arch_subdir=$(dirname "${model_subdir}")
        arch_name=$(basename "${arch_subdir}")

        device=$(get_device)

        python -W ignore metric_iqa.py \
            --input-dir "${input_dir}" \
            --target-dir "${target_dir}" \
            --result-file "${project_dir}" \
            --arch "${arch_name}" \
            --model "${model_name}" \
            --data "${data}" \
            --device "${device}" \
            --imgsz "${imgsz}" \
            ${resize} \
            "${metric_args[@]}" \
            ${use_gt_mean} \
            --backend "pyiqa" \
            "$@"
    done
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
