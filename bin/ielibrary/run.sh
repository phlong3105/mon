#!/bin/bash

echo "$HOSTNAME"

# Input
machine=$HOSTNAME
model=${1:-"zerodce"}
task=${2:-"train"}
train_data=${3:-"lol"}
predict_data=${4:-"lol"}
epoch=${5:-"100"}

read -e -i "$model"        -p "Model [zerodce, zerodce++]: " model
read -e -i "$task"         -p "Task [train, predict]: " task
read -e -i "$train_data"   -p "Train Data [lol]: " train_data
read -e -i "$predict_data" -p "Predict Data [lol]: " predict_data
read -e -i "$epoch"        -p "Epoch: " epoch

machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
model=$(echo $model | tr '[:upper:]' '[:lower:]')
task=$(echo $task | tr '[:upper:]' '[:lower:]')
train_data=$(echo $train_data | tr '[:upper:]' '[:lower:]')
predict_data=$(echo $predict_data | tr '[:upper:]' '[:lower:]')
epoch=$(($epoch))

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
zoo_dir="${root_dir}/zoo"

model_dir="${root_dir}/src/lib/${model}"
cd "${model_dir}" || exit

if [ "$task" == "train" ]; then
  if [ "$train_data" == "lol" ]; then
    low_data_dir="${root_dir}/data/lol/train/low/lol"
    high_data_dir="${root_dir}/data/lol/train/high/lol"
  fi
elif [ "$task" == "predict" ]; then
  if [ "$predict_data" == "lol" ]; then
      low_data_dir="${root_dir}/data/lol/val/low/lol"
      high_data_dir="${root_dir}/data/lol/val/high/lol"
  fi
fi

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$model" == "zerodce" ]; then
    python lowlight_train.py \
      --data "${low_data_dir}" \
      --lr 0.0001 \
      --weight-decay 0.0001 \
      --grad-clip-norm 0.1 \
      --epochs "${epoch}" \
      --train-batch-size 8 \
      --val-batch-size 4 \
      --num-workers 4 \
      --display-iter 10 \
      --checkpoints-iter 10 \
      --checkpoints-dir "${root_dir}/run/train/ielibrary/${model}-${train_data}" \
      --weights "${zoo_dir}/${model}/zerodce-lol.pth" \
      --load_pretrain false
  elif [ "$model" == "zerodce++" ]; then
    python lowlight_train.py \
      --data "${low_data_dir}" \
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
      --checkpoints-dir "${root_dir}/run/train/ielibrary/${model}-${train_data}" \
      --weights "${zoo_dir}/${model}/zerodce++-lol.pth" \
      --load_pretrain false
  fi
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  if [ "$model" == "zerodce" ]; then
    python lowlight_test.py \
      --data "${low_data_dir}" \
      --weights "${root_dir}/run/train/ielibrary/${model}-${train_data}/best.pt" \
      --output-dir "${root_dir}/run/predict/ielibrary/${model}/${predict_data}"
  elif [ "$model" == "zerodce++" ]; then
    python lowlight_test.py \
      --data "${low_data_dir}" \
      --weights "${root_dir}/run/train/ielibrary/${model}-${train_data}/best.pt" \
      --output-dir "${root_dir}/run/predict/ielibrary/${model}/${predict_data}"
  fi
fi

# Done
cd "${root_dir}" || exit
