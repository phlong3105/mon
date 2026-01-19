#!/bin/bash

# ==============================================================================
# SAFEGUARDS (Strict Mode)
# ==============================================================================
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error.
# -o pipefail: Capture errors even inside a pipeline (cmd1 | cmd2).
set -euo pipefail


# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
source ./utils.sh


# ==============================================================================
# VARIABLES
# ==============================================================================
# --- User Inputs ---
task=""
mode="predict"
arch=""
model=""
datasets=("")

# --- Paths ---
current_dir=$(pwd)
project_dir=$(dirname "${current_dir}")
root_dir=$(get_root_dir "$(pwd)")

# --- Derived Variables ---
data=$(printf "%s, " "${datasets[@]}")
data=${data%, }  # Remove trailing ", "

device=$(get_device)


# ==============================================================================
# EXECUTION
# ==============================================================================
echo "${HOSTNAME}"
clear

python -W ignore -m mon.tools.run_model \
    --root "${project_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --data "${data}" \
    --device "${device}" \
    --save \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    "$@"

echo "Running finished."
exit 0
