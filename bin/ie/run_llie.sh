#!/bin/bash

echo "$HOSTNAME"

# Constant
models=("zerodce" "zerodce++")
train_datasets=("lol")
predict_datasets=("dcim" "fusion" "lime" "lol" "mef" "npe" "sice" "vip" "vv")

# Input
machine=$HOSTNAME
model=${1:-"zerodce"}
task=${2:-"predict"}
train_data=${3:-"lol"}
predict_data=${4:-"all"}
project=${5:-"ie/llie"}
epoch=${6:-"100"}

read -e -i "$model"        -p "Model [${models[*]}]: " model
read -e -i "$task"         -p "Task [train, predict]: " task
read -e -i "$train_data"   -p "Train data [${train_datasets[*]}]: " train_data
read -e -i "$predict_data" -p "Predict data [all ${predict_datasets[*]}]: " predict_data
read -e -i "$project"      -p "Project: " project
read -e -i "$epoch"        -p "Epoch: " epoch

machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
model=$(echo $model | tr '[:upper:]' '[:lower:]')
task=$(echo $task | tr '[:upper:]' '[:lower:]')
train_data=$(echo $train_data | tr '[:upper:]' '[:lower:]')
predict_data=$(echo $predict_data | tr '[:upper:]' '[:lower:]')
project=$(echo $project | tr '[:upper:]' '[:lower:]')
epoch=$(($epoch))

# Check Input
if [[ ! "${models[*]}" =~ $model ]]; then
  echo "$model is not in [${models[*]}]"
fi
if [[ ! "${train_datasets[*]}" =~ $train_data ]]; then
  echo "$train_data is not in [${train_datasets[*]}]"
fi
# if [[ ! "${predict_datasets[*]}" =~ $predict_data ]]; then
#   echo "$predict_data is not in [${predict_datasets[*]}]"
# fi

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
model_dir="${root_dir}/src/lib/${model}"
cd "${model_dir}" || exit

if [ "$predict_data" == "all" ]; then
  declare -a predict_data=()
  for i in ${!predict_datasets[@]}; do
      predict_data[$i]="${predict_datasets[$i]}"
  done
else
  declare -a predict_data=()
  predict_data+=($predict_data)
fi
# echo "${predict_data[*]}"
  
declare -a low_data_dirs=()
declare -a high_data_dirs=()
if [ "$task" == "train" ]; then
  if [ "$train_data" == "lol" ]; then
      low_data_dirs+=("${root_dir}/data/lol/train/low/lol")
      high_data_dirs+=("${root_dir}/data/lol/train/high/lol")
  fi
elif [ "$task" == "predict" ]; then
  for d in "${predict_data[@]}"; do
    if [ "$d" == "dcim" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/dcim")
        high_data_dirs+=("")
    fi
    if [ "$d" == "fusion" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/fusion")
        high_data_dirs+=("")
    fi
    if [ "$d" == "lime" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/lime")
        high_data_dirs+=("")
    fi
    if [ "$d" == "lol" ]; then
        low_data_dirs+=("${root_dir}/data/lol/val/low/lol")
        high_data_dirs+=("${root_dir}/data/lol/val/high/lol")
    fi
    if [ "$d" == "mef" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/mef")
        high_data_dirs+=("")
    fi
    if [ "$d" == "npe" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/npe")
        high_data_dirs+=("")
    fi
    if [ "$d" == "sice" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/sice")
        high_data_dirs+=("")
    fi
    if [ "$d" == "vip" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/vip")
        high_data_dirs+=("")
    fi
    if [ "$d" == "vv" ]; then
        low_data_dirs+=("${root_dir}/data/lol/train/low/vv")
        high_data_dirs+=("")
    fi
  done
fi
# echo "${low_data_dirs[*]}"

train_dir="${root_dir}/run/train/${project}/${model}-${train_data}"
train_weights="${root_dir}/run/train/${project}/${model}-${train_data}/best.pt"
zoo_weights="${root_dir}/zoo/${model}/${model}-${train_data}.pth"
if [ -f "$train_weights" ]; then
    weights=${train_weights}
else
    weights="${zoo_weights}"
fi

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$model" == "zerodce" ]; then
    python lowlight_train.py \
      --data "${low_data_dirs[i]}" \
      --lr 0.0001 \
      --weight-decay 0.0001 \
      --grad-clip-norm 0.1 \
      --epochs "${epoch}" \
      --train-batch-size 8 \
      --val-batch-size 4 \
      --num-workers 4 \
      --display-iter 10 \
      --checkpoints-iter 10 \
      --checkpoints-dir "${train_dir}" \
      --weights "${zoo_weights}" \
      --load_pretrain false
  elif [ "$model" == "zerodce++" ]; then
    python lowlight_train.py \
      --data "${low_data_dirs[i]}" \
      --lr 0.0001 \
      --weight-decay 0.0001 \
      --grad-clip-norm 0.1 \
      --scale-factor 1 \
      --epochs "${epoch}" \
      --train-batch-size 8 \
      --val-batch-size 4 \
      --num-workers 4 \
      --display-iter 10 \
      --checkpoints-iter 10 \
      --checkpoints-dir "${train_dir}" \
      --weights "${zoo_weights}" \
      --load_pretrain false
  fi
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  for (( i=0; i<${#predict_data[@]}; i++ )); do
    predict_dir="${root_dir}/run/predict/${project}/${model}/${predict_data[$i]}"
    # echo -e "${low_data_dirs[$i]}"
    if [ "$model" == "zerodce" ]; then
      python lowlight_test.py \
        --data "${low_data_dirs[$i]}" \
        --weights "${weights}" \
        --output-dir "${predict_dir}"
    elif [ "$model" == "zerodce++" ]; then
      python lowlight_test.py \
        --data "${low_data_dirs[$i]}" \
        --weights "${weights}" \
        --output-dir "${predict_dir}"
    fi
  done
fi

# Done
cd "${root_dir}" || exit
