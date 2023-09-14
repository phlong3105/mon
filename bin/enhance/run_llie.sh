#!/bin/bash

echo "$HOSTNAME"

# Constant
models=(
  "iat"          # https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance
  "lcdpnet"      # https://github.com/onpix/LCDPNet
  "llflow"       # https://github.com/wyf0912/LLFlow
  "retinexdip"   # https://github.com/zhaozunjin/RetinexDIP
  "ruas"         # https://github.com/KarelZhang/RUAS
  "sci"          # https://github.com/vis-opt-group/SCI
  "sgz"          #
  "snr"          # https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
  "uretinexnet"  # https://github.com/AndersonYong/URetinex-Net
  "zerodce"      #
  "zerodce++"    #
)
train_datasets=("lol")
predict_datasets=("dcim" "fusion" "lime" "lol" "mef" "npe" "sice" "vip" "vv")

# Input
machine=$HOSTNAME
model=${1:-"zerodce"}
task=${2:-"predict"}
train_data=${3:-"lol"}
predict_data=${4:-"all"}
project=${5:-"enhance/llie"}
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
model_dir="${root_dir}/src/lib/vision/enhance/${model}"
cd "${model_dir}" || exit

if [ "$predict_data" == "all" ]; then
  declare -a predict_data=()
  for i in ${!predict_datasets[@]}; do
    predict_data[$i]="${predict_datasets[$i]}"
  done
else
  declare -a predict_data=($predict_data)
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
      low_data_dirs+=("${root_dir}/data/lol/train/low/lol")
      high_data_dirs+=("${root_dir}/data/lol/train/high/lol")
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
train_weights_pt="${root_dir}/run/train/${project}/${model}-${train_data}/best.pt"
train_weights_pth="${root_dir}/run/train/${project}/${model}-${train_data}/best.pth"
train_weights_ckpt="${root_dir}/run/train/${project}/${model}-${train_data}/best.ckpt"
zoo_weights_pt="${root_dir}/zoo/vision/enhance/${model}/${model}-${train_data}.pt"
zoo_weights_pth="${root_dir}/zoo/vision/enhance/${model}/${model}-${train_data}.pth"
zoo_weights_ckpt="${root_dir}/zoo/vision/enhance/${model}/${model}-${train_data}.ckpt"
if [ -f "$train_weights_pt" ]; then
  weights=${train_weights_pt}
elif [ -f "$train_weights_pth" ]; then
  weights="${train_weights_pth}"
elif [ -f "$train_weights_ckpt" ]; then
  weights="${train_weights_ckpt}"
elif [ -f "$zoo_weights_pt" ]; then
  weights="${zoo_weights_pt}"
elif [ -f "$zoo_weights_pth" ]; then
  weights="${zoo_weights_pth}"
elif [ -f "$zoo_weights_ckpt" ]; then
  weights="${zoo_weights_ckpt}"
fi

if [ -f "$zoo_weights_ckpt" ]; then
  zoo_weights="${zoo_weights_ckpt}"
elif [ -f "$zoo_weights_pth" ]; then
  zoo_weights="${zoo_weights_pth}"
else
  zoo_weights="${zoo_weights_pt}"
fi

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  # IAT
  if [ "$model" == "iat" ]; then
    echo -e "\nI have not implemented training script for IAT."
  # LCDPNet
  elif [ "$model" == "lcdpnet" ]; then
    python src/train.py \
      name="lcdpnet-lol" \
      num_epoch=${epoch} \
      log_every=2000 \
      valid_every=20
  # LLFlow
  elif [ "$model" == "llflow" ]; then
    echo -e "\nI have not implemented training script for LLFlow."
  # RetinexDIP
  elif [ "$model" == "retinexdip" ]; then
    echo -e "\nRetinexNet should be run in prediction mode only"
    python retinexdip.py \
      --data "${low_data_dirs[i]}" \
      --weights "${zoo_weights}" \
      --image-size 512 \
      --output-dir "${train_dir}"
  # RUAS
  elif [ "$model" == "ruas" ]; then
    python train.py \
      --data "${low_data_dirs[i]}" \
      --weights "${zoo_weights}" \
      --load-pretrain false \
      --epoch "${epoch}" \
      --batch-size 1 \
      --report-freq 50 \
      --gpu 0 \
      --seed 2 \
      --checkpoints-dir "${train_dir}"
  # SCI
  elif [ "$model" == "sci" ]; then
    python train.py \
      --data "${low_data_dirs[i]}" \
      --weights "${zoo_weights}" \
      --load-pretrain false \
      --batch-size 1 \
      --epochs "${epoch}" \
      --lr 0.0003 \
      --stage 3 \
      --cuda true \
      --gpu 0 \
      --seed 2 \
      --checkpoints-dir "${train_dir}"
  # SGZ
  elif [ "$model" == "sgz" ]; then
    python train.py \
      --data "${low_data_dirs[i]}" \
      --weights "${zoo_weights}" \
      --load-pretrain false \
      --image-size 512 \
      --lr 0.0001 \
      --weight-decay 0.0001 \
      --grad-clip-norm 0.1 \
      --epochs "${epoch}" \
      --train-batch-size 6 \
      --val-batch-size 8 \
      --num-workers 4 \
      --display-iter 10 \
      --scale-factor 1 \
      --num-of-SegClass 21 \
      --conv-type "dsc" \
      --patch-size 4 \
      --exp-level 0.6 \
      --checkpoints-iter 10 \
      --checkpoints-dir "${train_dir}"
  # SGZ
  elif [ "$model" == "snr" ]; then
    echo -e "\nI have not implemented training script for SNR-Aware."
  # URetinex-Net
  elif [ "$model" == "uretinexnet" ]; then
    echo -e "\nI have not implemented training script for URetinex-Net."
  # Zero-DCE
  elif [ "$model" == "zerodce" ]; then
    python lowlight_train.py \
      --data "${low_data_dirs[i]}" \
      --weights "${zoo_weights}" \
      --load-pretrain false \
      --lr 0.0001 \
      --weight-decay 0.0001 \
      --grad-clip-norm 0.1 \
      --epochs "${epoch}" \
      --train-batch-size 8 \
      --val-batch-size 4 \
      --num-workers 4 \
      --display-iter 10 \
      --checkpoints-iter 10 \
      --checkpoints-dir "${train_dir}"
  # Zero-DCE++
  elif [ "$model" == "zerodce++" ]; then
    python lowlight_train.py \
      --data "${low_data_dirs[i]}" \
      --weights "${zoo_weights}" \
      --load-pretrain false \
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
      --checkpoints-dir "${train_dir}"
  fi
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  # LCDPNet
  if [ "$model" == "lcdpnet" ]; then
    python src/test.py \
      checkpoint_path="${weights}" \
      +image_size=512
  else
    for (( i=0; i<${#predict_data[@]}; i++ )); do
      predict_dir="${root_dir}/run/predict/${project}/${model}/${predict_data[$i]}"
      # IAT
      if [ "$model" == "iat" ]; then
          python IAT_enhance/predict.py \
            --data "${low_data_dirs[i]}" \
            --exposure-weights "${root_dir}/zoo/vision/enhance/${model}/iat-exposure.pth" \
            --enhance-weights "${root_dir}/zoo/vision/enhance/${model}/iat-lol.pth" \
            --image-size 512 \
            --normalize \
            --task "enhance" \
            --output-dir "${predict_dir}"
      # LLFlow
      elif [ "$model" == "llflow" ]; then
        python code/test_unpaired_v2.py \
          --data "${low_data_dirs[i]}" \
          --weights "${root_dir}/zoo/vision/enhance/${model}/llflow-lol-smallnet.pth" \
          --image-size 512 \
          --output-dir "${predict_dir}" \
          --opt "code/confs/LOL_smallNet.yml" \
          --name "unpaired"
      # RetinexDIP
      elif [ "$model" == "retinexdip" ]; then
        python retinexdip.py \
          --data "${low_data_dirs[i]}" \
          --weights "${weights}" \
          --image-size 512 \
          --output-dir "${predict_dir}"
      # RUAS
      elif [ "$model" == "ruas" ]; then
        python test.py \
          --data "${low_data_dirs[$i]}" \
          --weights "${weights}" \
          --image-size 512 \
          --gpu 0 \
          --seed 2 \
          --output-dir "${predict_dir}"
      # SCI
      elif [ "$model" == "sci" ]; then
        python test.py \
          --data "${low_data_dirs[i]}" \
          --weights "${zoo_weights}" \
          --image-size 512 \
          --gpu 0 \
          --seed 2 \
          --output-dir "${predict_dir}"
      # SGZ
      elif [ "$model" == "sgz" ]; then
        python test.py \
          --data "${low_data_dirs[i]}" \
          --weights "${zoo_weights}" \
          --image-size 512 \
          --output-dir "${predict_dir}"
      # SGZ
      elif [ "$model" == "snr" ]; then
        python predict.py \
          --data "${low_data_dirs[i]}" \
          --weights "${root_dir}/zoo/vision/enhance/${model}/snr-lolv1.pth" \
          --opt "./options/test/LOLv1.yml" \
          --image-size 512 \
          --output-dir "${predict_dir}"
      # URetinex-Net
      elif [ "$model" == "uretinexnet" ]; then
        python test.py \
          --data "${low_data_dirs[i]}" \
          --decom-model-low-weights "${root_dir}/zoo/vision/enhance/${model}/${train_data}/init_low-lol.pth" \
          --unfolding-model-weights "${root_dir}/zoo/vision/enhance/${model}/${train_data}/unfolding-lol.pth" \
          --adjust-model-weights "${root_dir}/zoo/vision/enhance/${model}/${train_data}/L_adjust-lol.pth" \
          --image-size 512 \
          --ratio 5 \
          --output-dir "${predict_dir}"
      # Zero-DCE
      elif [ "$model" == "zerodce" ]; then
        python lowlight_test.py \
          --data "${low_data_dirs[$i]}" \
          --weights "${weights}" \
          --image-size 512 \
          --output-dir "${predict_dir}"
      # Zero-DCE++
      elif [ "$model" == "zerodce++" ]; then
        python lowlight_test.py \
          --data "${low_data_dirs[$i]}" \
          --weights "${weights}" \
          --image-size 512 \
          --output-dir "${predict_dir}"
      fi
    done
  fi
fi

# Done
cd "${root_dir}" || exit
