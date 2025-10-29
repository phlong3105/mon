#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
arch="dav2"
model="dav2_vitb"
datasets=(
    ### Unpaired
    #"dicm"
    #"lime"
    #"mef"
    #"npe"
    #"vv"
    ### LOLs
    #"lolv1"
    #"lolv2real"
    #"lolv2syn"
    ### LSRW
    #"lsrw"
    ### SICE
    #"sice"
    ### FiveK
    #"fiveka"
    #"fivekb"
    #"fivekc"
    #"fivekd"
    #"fiveke"
    ### High-Level
    #"darkface"
    #"exdark"
    #"lolistreettest"
    #"lolistreetval"
    #"nightcity"
)

# ----- Directory & File -----
current_dir=$(pwd)
project_dir=$(dirname "${current_dir}")
root_dir=$(get_root_dir "$(pwd)")
run_dir="${root_dir}/shared/mon/mon_run"

declare -A input_subdirs=(
    ["lolv2real"]="lolv2/real/test/image"
    ["lolv2syn"]="lolv2/syn/test/image"
    ["fiveka"]="fivek/test/image"
    ["fivekb"]="fivek/test/image"
    ["fivekc"]="fivek/test/image"
    ["fivekd"]="fivek/test/image"
    ["fiveke"]="fivek/test/image"
    ["sice"]="sice/sice/test/image"
    ["lolistreetval"]="lolistreet/val/image"
    ["lolistreettest"]="lolistreet/test/image"
)
declare -A target_subdirs=(
    ["lolv2real"]="lolv2/real/test/ref"
    ["lolv2syn"]="lolv2/syn/test/ref"
    ["fiveka"]="fivek/test/ref_a"
    ["fivekb"]="fivek/test/ref_b"
    ["fivekc"]="fivek/test/ref_c"
    ["fivekd"]="fivek/test/ref_d"
    ["fiveke"]="fivek/test/ref_e"
    ["sice"]="sice/sice/test/ref"
    ["lolistreetval"]="lolistreet/val/ref"
    ["lolistreettest"]="lolistreet/test/ref"
)

# ----- Main -----
cd "${run_dir}" || exit

for data in "${datasets[@]}"; do
    # Input
    input_subdir="${input_subdirs[$data]:-${data}/test/image}"
    input_dir="${project_dir}/data/${input_subdir}_${model}"
    check_dir "${input_dir}"

    # Target
    target_subdir="${target_subdirs[$data]:-${data}/test/ref}"
    target_dir="${project_dir}/data/${target_subdir}_${model}"
    check_dir "${target_dir}"

    device=$(get_device)

    # Run evaluation
    python -W ignore metric_depth.py \
        --input-dir "${input_dir}" \
        --target-dir "${target_dir}" \
        --result-file "${project_dir}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${data}" \
        --device "${device}" \
        --imgsz 512 \
        --use-color \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
