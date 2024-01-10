#!/bin/bash
echo "$HOSTNAME"

## CONSTANTS
# Task
tasks=(
  "deblur"
  "dehaze"
  "denoise"
  "derain"
  "desnow"
  "les"
  "llie"
)
# Models
deblur_models=(
  "finet"         # https://github.com/phlong3105/mon
  "hinet"         # https://github.com/megvii-model/HINet
  "uformer"       # https://github.com/ZhendongWang6/Uformer
)
dehaze_models=(
  "finet"         # https://github.com/phlong3105/mon
  "zid"           # https://github.com/liboyun/ZID
)
denoise_models=(
  "finet"         # https://github.com/phlong3105/mon
  "hinet"         # https://github.com/megvii-model/HINet
  "uformer"       # https://github.com/ZhendongWang6/Uformer
)
derain_models=(
  "finet"         # https://github.com/phlong3105/mon
  "hinet"         # https://github.com/megvii-model/HINet
  "ipt"           # https://github.com/huawei-noah/Pretrained-IPT
  "transweather"  #
  "uformer"       # https://github.com/ZhendongWang6/Uformer
)
desnow_models=(
  "finet"         # https://github.com/phlong3105/mon
  "hinet"         # https://github.com/megvii-model/HINet
)
les_models=(
  "jin2022"       # https://github.com/jinyeying/night-enhancement
)
llie_models=(
  "enlightengan"  # https://github.com/arsenyinfo/EnlightenGAN-inference
  "gcenet"        # https://github.com/phlong3105/mon
  "gcenetv2"      # https://github.com/phlong3105/mon
  "iat"           # https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance
  "kind"          # https://github.com/zhangyhuaee/KinD
  "kind++"        # https://github.com/zhangyhuaee/KinD_plus
  "lcdpnet"       # https://github.com/onpix/LCDPNet
  "lime"          # https://github.com/pvnieo/Low-light-Image-Enhancement
  "llflow"        # https://github.com/wyf0912/LLFlow
  "mbllen"        # https://github.com/Lvfeifan/MBLLEN
  "pie"           # https://github.com/DavidQiuChao/PIE
  "retinexdip"    # https://github.com/zhaozunjin/RetinexDIP
  "retinexnet"    # https://github.com/weichen582/RetinexNet
  "ruas"          # https://github.com/KarelZhang/RUAS
  "sci"           # https://github.com/vis-opt-group/SCI
  "sgz"           #
  "snr"           # https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
  "stablellve"    # https://github.com/zkawfanx/StableLLVE
  "uretinexnet"   # https://github.com/AndersonYong/URetinex-Net
  "utvnet"        # https://github.com/CharlieZCJ/UTVNet
  "zeroadce"      # https://github.com/phlong3105/mon
  "zerodce"       # https://github.com/Li-Chongyi/Zero-DCE
  "zerodce++"     # https://github.com/Li-Chongyi/Zero-DCE_extension
)
# Datasets
deblur_datasets=()
dehaze_datasets=(
  "dense-haze"
  "i-haze"
  "light-effect"
  "nh-haze"
  "o-haze"
)
denoise_datasets=()
derain_datasets=(
  "gt-rain"
  "rain100h"
  "rain100l"
)
desnow_datasets=(
  "gt-snow"
)
les_datasets=(
  "ledlight"
  "light-effect"
)
llie_datasets=(
  "darkcityscapes"
  "darkface"
  "dicm"
  "exdark"
  "fivek-c"
  "fivek-e"
  "fusion"
  "lime"
  "lol-v1"
  "lol-v2-real"
  "lol-v2-syn"
  "mef"
  "npe"
  "sice"
  "sice-grad"
  "sice-mix"
  "sice-zerodce"
  "vv"
)


## READ INPUTS
# Command Args
host=$HOSTNAME
task=${1:-"none"}
run=${2:-"predict"}
train=${3:-"none"}
predict=${4:-"none"}
model=${5:-"none"}
variant=${6:-"none"}
suffix=${7:-"none"}
project=${8:-"vision/enhance/"}
epochs=${9:-100}
use_data_dir=${10:-"no"}
checkpoint=${11:-"best"}

# User Input
read -e -i "$task" -p "Task [${tasks[*]}]: " task
read -e -i "$run"  -p "Task [train, predict, evaluate, plot]: " run
task=$(echo $task | tr '[:upper:]' '[:lower:]')
run=$(echo $run | tr '[:upper:]' '[:lower:]')
if [ "$task" == "deblur" ]; then
  read -e -i "$train"   -p "Train data [${deblur_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${deblur_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${deblur_models[*]}]: " model
elif [ "$task" == "dehaze" ]; then
  read -e -i "$train"   -p "Train data [${dehaze_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${dehaze_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${dehaze_models[*]}]: " model
elif [ "$task" == "denoise" ]; then
  read -e -i "$train"   -p "Train data [${denoise_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${denoise_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${denoise_models[*]}]: " model
elif [ "$task" == "derain" ]; then
  read -e -i "$train"   -p "Train data [${derain_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${derain_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${derain_models[*]}]: " model
elif [ "$task" == "desnow" ]; then
  read -e -i "$train"   -p "Train data [${desnow_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${desnow_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${desnow_models[*]}]: " model
elif [ "$task" == "les" ]; then
  read -e -i "$train"   -p "Train data [${les_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${les_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${les_models[*]}]: " model
elif [ "$task" == "llie" ]; then
  read -e -i "$train"   -p "Train data [${llie_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${llie_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${llie_models[*]}]: " model
else
  echo -e "\nWrong task"
  exit 1
fi
project="${project}${task}"
read -e -i "$variant" -p "Variant: " variant
read -e -i "$suffix"  -p "Suffix: " suffix
read -e -i "$project" -p "Project: " project
if [ "$run" == "train" ]; then
  read -e -i "$epochs"  -p "Epochs: " epochs
elif [ "$run" == "predict" ]; then
  read -e -i "$use_data_dir" -p "Use data_dir [yes, no]: " use_data_dir
  read -e -i "$checkpoint"   -p "Checkpoint type [best, last]: " checkpoint
elif [ "$run" == "evaluate" ] || [ "$run" == "plot" ]; then
  read -e -i "$use_data_dir" -p "Use data_dir [yes, no]: " use_data_dir
fi

# Check Args
echo -e "\n"
host=$(echo $host | tr '[:upper:]' '[:lower:]')
train=$(echo $train | tr '[:upper:]' '[:lower:]')
train=($(echo $train | tr ',' '\n'))
predict=$(echo $predict | tr '[:upper:]' '[:lower:]')
predict=($(echo $predict | tr ',' '\n'))
model=$(echo $model | tr '[:upper:]' '[:lower:]')
model=($(echo $model | tr ',' '\n'))
variant=$(echo $variant | tr '[:upper:]' '[:lower:]')
variant=($(echo "$variant" | tr ',' '\n'))
suffix=$(echo $suffix | tr '[:upper:]' '[:lower:]')
project=$(echo $project | tr '[:upper:]' '[:lower:]')


## INITIALIZATION
# Directories
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
data_dir="${root_dir}/data"
run_dir="${root_dir}/run"
model_dir="${root_dir}/src/lib/vision/enhance/${task}"
deblur_dir="${root_dir}/src/lib/vision/enhance/deblur"
dehaze_dir="${root_dir}/src/lib/vision/enhance/dehaze"
denoise_dir="${root_dir}/src/lib/vision/enhance/denoise"
derain_dir="${root_dir}/src/lib/vision/enhance/derain"
desnow_dir="${root_dir}/src/lib/vision/enhance/desnow"
les_dir="${root_dir}/src/lib/vision/enhance/les"
llie_dir="${root_dir}/src/lib/vision/enhance/llie"
universal_dir="${root_dir}/src/lib/vision/enhance/universal"

# Datasets
declare -a datasets=()
declare -a input_dirs=()
declare -a target_dirs=()
if [ "$run" == "train" ]; then
  datasets=("${train[@]}")
  split="train"
elif [ "$run" == "predict" ]; then
  datasets=("${predict[@]}")
  split="test"
fi
for d in "${datasets[@]}"; do
  # De-Hazing
  if [ "$d" == "dense-haze" ]; then
    input_dirs+=("${data_dir}/dehaze/${split}/${d}/haze")
    target_dirs+=("${data_dir}/dehaze/${split}/${d}/clear")
  fi
  if [ "$d" == "i-haze" ]; then
    input_dirs+=("${data_dir}/dehaze/${split}/${d}/haze")
    target_dirs+=("${data_dir}/dehaze/${split}/${d}/clear")
  fi
  if [ "$d" == "nh-haze" ]; then
    input_dirs+=("${data_dir}/dehaze/${split}/${d}/haze")
    target_dirs+=("${data_dir}/dehaze/${split}/${d}/clear")
  fi
  if [ "$d" == "o-haze" ]; then
    input_dirs+=("${data_dir}/dehaze/${split}/${d}/haze")
    target_dirs+=("${data_dir}/dehaze/${split}/${d}/clear")
  fi

  # De-Raining
  if [ "$d" == "gt-rain" ]; then
    input_dirs+=("${data_dir}/derain/${split}/${d}/rain")
    target_dirs+=("${data_dir}/derain/${split}/${d}/clear")
  fi
  if [ "$d" == "rain100h" ]; then
    input_dirs+=("${data_dir}/derain/${split}/${d}/rain")
    target_dirs+=("${data_dir}/derain/${split}/${d}/clear")
  fi
  if [ "$d" == "rain100l" ]; then
    input_dirs+=("${data_dir}/derain/${split}/${d}/rain")
    target_dirs+=("${data_dir}/derain/${split}/${d}/clear")
  fi

  # De-Snowing
  if [ "$d" == "gt-snow" ]; then
    input_dirs+=("${data_dir}/desnow/train/${d}/snow")
    target_dirs+=("${data_dir}/desnow/train/${d}/clear")
  fi

  # LES
  if [ "$d" == "ledlight" ]; then
    input_dirs+=("${data_dir}/les/test/${d}/light-effects")
    target_dirs+=("${data_dir}/les/test/${d}/clear")
  fi
  if [ "$d" == "light-effect" ]; then
    input_dirs+=("${data_dir}/les/train/${d}/light-effects")
    target_dirs+=("")
  fi

  # LLIE
  if [ "$d" == "darkcityscapes" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("${data_dir}/llie/test/${d}/high")
  fi
  if [ "$d" == "darkface" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "dicm" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "exdark" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "fivek-c" ]; then
    input_dirs+=("${data_dir}/llie/train/${d}/low")
    target_dirs+=("${data_dir}/llie/train/${d}/high")
  fi
  if [ "$d" == "fivek-e" ]; then
    input_dirs+=("${data_dir}/llie/train/${d}/low")
    target_dirs+=("${data_dir}/llie/train/${d}/high")
  fi
  if [ "$d" == "fusion" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "lime" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "lol-v1" ]; then
    input_dirs+=("${data_dir}/llie/${split}/${d}/low")
    target_dirs+=("${data_dir}/llie/${split}/${d}/high")
  fi
  if [ "$d" == "lol-v2-real" ]; then
    input_dirs+=("${data_dir}/llie/${split}/${d}/low")
    target_dirs+=("${data_dir}/llie/${split}/${d}/high")
  fi
  if [ "$d" == "lol-v2-syn" ]; then
    input_dirs+=("${data_dir}/llie/${split}/${d}/low")
    target_dirs+=("${data_dir}/llie/${split}/${d}/high")
  fi
  if [ "$d" == "mef" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "npe" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
  if [ "$d" == "sice" ]; then
    if [ "$run" == "train" ]; then
      input_dirs+=("${data_dir}/llie/train/sice-part1/low")
      target_dirs+=("${data_dir}/llie/train/sice-part1/high")
    else
      input_dirs+=("${data_dir}/llie/train/sice-part2/low")
      target_dirs+=("${data_dir}/llie/train/sice-part2/high")
    fi
  fi
  if [ "$d" == "sice-grad" ]; then
    input_dirs+=("${data_dir}/llie/train/${d}/low")
    target_dirs+=("${data_dir}/llie/train/${d}/high")
  fi
  if [ "$d" == "sice-mix" ]; then
    input_dirs+=("${data_dir}/llie/train/${d}/low")
    target_dirs+=("${data_dir}/llie/train/${d}/high")
  fi
  if [ "$d" == "sice-zerodce" ]; then
    input_dirs+=("${data_dir}/llie/train/${d}/low")
    target_dirs+=("${data_dir}/llie/train/${d}/high")
  fi
  if [ "$d" == "vv" ]; then
    input_dirs+=("${data_dir}/llie/test/${d}/low")
    target_dirs+=("")
  fi
done


## TRAIN
if [ "$run" == "train" ]; then
  echo -e "\nTraining"
  for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#variant[@]}; j++ )); do
      # Model initialization
      if [ "${variant[j]}" != "none" ] && [ "${variant[j]}" != "" ]; then
        model_variant="${model[i]}-${variant[j]}"
      else
        model_variant="${model[i]}"
      fi
      if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
        name="${model_variant}-${train[0]}-${suffix}"
      else
        name="${model_variant}-${train[0]}"
      fi
      config="${model[i]}_${train[0]/-/_}"
      # Weights initialization
      train_dir="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train[0]}"
      zoo_weights_pt="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train[0]}.pt"
      zoo_weights_pth="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train[0]}.pth"
      zoo_weights_ckpt="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train[0]}.ckpt"
      if [ -f "$zoo_weights_ckpt" ]; then
        weights="${zoo_weights_ckpt}"
      elif [ -f "$zoo_weights_pth" ]; then
        weights="${zoo_weights_pth}"
      else
        weights="${zoo_weights_pt}"
      fi

      # De-Hazing
      # ZID
      if [ "${model[i]}" == "zid" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs 500

      # De-Raining
      # IPT
      elif [ "${model[i]}" == "ipt" ]; then
        cd "${derain_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for IPT."
        # python -W ignore main.py \
        #   --dir_data "${input_dirs[j]}" \
        #   --pretrain "${weights}" \
        #   --load-pretrain false \
        #   --lr 0.0001 \
        #   --weight-decay 0.0001 \
        #   --grad-clip-norm 0.1 \
        #   --scale-factor 1 \
        #   --epochs "$epochs" \
        #   --train-batch-size 8 \
        #   --val-batch-size 4 \
        #   --num-workers 4 \
        #   --display-iter 10 \
        #   --checkpoints-iter 10 \
        #   --checkpoints-dir "${train_dir}"
      # TransWeather
      elif [ "${model[i]}" == "transweather" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs" \
          --strategy "ddp_find_unused_parameters_true"

      # LES
      # Jin2022
      elif [ "${model[i]}" == "jin2022" ]; then
        cd "${les_dir}/${model[i]}" || exit
        python -W ignore demo_all.py \
          --data "${input_dirs[k]}" \
          --image-size 512 \
          --output-dir "${train_dir}/visual" \
          --checkpoint-dir "${train_dir}"

      # LLIE
      # EnlightenGAN
      elif [ "${model[i]}" == "enlightengan" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for EnlightenGAN."
      # GCENet
      elif [ "${model[i]}" == "gcenet" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --config "${config}" \
          --root "${run_dir}/train" \
          --project "${project}/${model[i]}" \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # GCENetV2
      elif [ "${model[i]}" == "gcenetv2" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --config "${config}" \
          --root "${run_dir}/train" \
          --project "${project}/${model[i]}" \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # IAT
      elif [ "${model[i]}" == "iat" ]; then
        cd "${${llie_dir}/${model[i]}}" || exit
        python -W ignore train_lol_v1_patch.py \
          --input-train "${input_dirs[j]}" \
          --input-val "${input_dirs[j]}" \
          --batch-size 8 \
          --lr 0.0002 \
          --weight-decay 0.0004 \
          --epochs "$epochs" \
          --gpu 0 \
          --display-iter 10 \
          --checkpoints-iter 10 \
          --checkpoints-dir "${train_dir}"
      # KinD
      elif [ "${model[i]}" == "kind" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for KinD."
      # KinD++
      elif [ "${model[i]}" == "kind++" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for KinD++."
      # LCDPNet
      elif [ "${model[i]}" == "lcdpnet" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore src/train.py \
          name="lcdpnet-lol" \
          num_epoch="$epochs" \
          log_every=2000 \
          valid_every=20
      # LLFlow
      elif [ "${model[i]}" == "llflow" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for LLFlow."
      # LIME
      elif [ "${model[i]}" == "lime" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nLIME does not need any training."
      # MBLLEN
      elif [ "${model[i]}" == "mbllen" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for MBLLEN."
      # PIE
      elif [ "${model[i]}" == "pie" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nPIE does not need any training."
      # RetinexDIP
      elif [ "${model[i]}" == "retinexdip" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore retinexdip.py \
          --input-dir "${input_dirs[j]}" \
          --output-dir "${train_dir}" \
          --weights "${root_dir}/zoo/vision/enhance/llie/retinexdip/retinexdip-lol.pt" \
          --image-size 512
      # RetinexNet
      elif [ "${model[i]}" == "retinexnet" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore train.py \
          --input-low "${input_dirs[j]}" \
          --input-high "${target_dirs[j]}" \
          --gpu 0 \
          --epochs "$epochs" \
          --batch-size 16 \
          --patch-size 96 \
          --lr 0.001 \
          --checkpoints-dir "${train_dir}"
      # RUAS
      elif [ "${model[i]}" == "ruas" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore train.py \
          --input-dir "${input_dirs[j]}" \
          --weights "${weights}" \
          --load-pretrain false \
          --epoch "$epochs" \
          --batch-size 1 \
          --report-freq 50 \
          --gpu 0 \
          --seed 2 \
          --checkpoints-dir "${train_dir}"
      # SCI
      elif [ "${model[i]}" == "sci" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore train.py \
          --input-dir "${input_dirs[j]}" \
          --weights "${weights}" \
          --load-pretrain false \
          --batch-size 1 \
          --epochs "$epochs" \
          --lr 0.0003 \
          --stage 3 \
          --cuda true \
          --gpu 0 \
          --seed 2 \
          --checkpoints-dir "${train_dir}"
      # SGZ
      elif [ "${model[i]}" == "sgz" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore train.py \
          --input-dir "${input_dirs[j]}" \
          --weights "${root_dir}/zoo/vision/enhance/llie/sgz/sgz-lol.pt" \
          --load-pretrain false \
          --image-size 512 \
          --lr 0.0001 \
          --weight-decay 0.0001 \
          --grad-clip-norm 0.1 \
          --epochs "$epochs" \
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
      # SNR-Aware
      elif [ "${model[i]}" == "snr" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for SNR-Aware."
      # StableLLVE
      elif [ "${model[i]}" == "stablellve" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore train.py \
          --input-dir "${input_dirs[j]}" \
          --epochs "$epochs" \
          --batch-size 1 \
          --lr 0.0001 \
          --weight 20 \
          --gpu 0 \
          --log-dir "${train_dir}" \
          --checkpoint-dir "${train_dir}"
      # URetinex-Net
      elif [ "${model[i]}" == "uretinexnet" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for URetinex-Net."
      # UTVNet
      elif [ "${model[i]}" == "utvnet" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        echo -e "\nI have not prepared the training script for UTVNet."
      # Zero-ADCE
      elif [ "${model[i]}" == "zeroadce" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --config "${config}" \
          --root "${run_dir}/train" \
          --project "${project}/${model[i]}" \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # Zero-DCE
      elif [ "${model[i]}" == "zerodce" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore lowlight_train.py \
          --input-dir "${input_dirs[j]}" \
          --weights "${root_dir}/zoo/vision/enhance/llie/zerodce/best.pth" \
          --load-pretrain false \
          --lr 0.0001 \
          --weight-decay 0.0001 \
          --grad-clip-norm 0.1 \
          --epochs "$epochs" \
          --train-batch-size 8 \
          --val-batch-size 4 \
          --num-workers 4 \
          --display-iter 10 \
          --checkpoints-iter 10 \
          --checkpoints-dir "${train_dir}"
      # Zero-DCE++
      elif [ "${model[i]}" == "zerodce++" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore lowlight_train.py \
          --input-dir "${input_dirs[j]}" \
          --weights "${root_dir}/zoo/vision/enhance/llie/zerodce++/best.pth" \
          --load-pretrain false \
          --lr 0.0001 \
          --weight-decay 0.0001 \
          --grad-clip-norm 0.1 \
          --scale-factor 1 \
          --epochs "$epochs" \
          --train-batch-size 8 \
          --val-batch-size 4 \
          --num-workers 4 \
          --display-iter 10 \
          --checkpoints-iter 10 \
          --checkpoints-dir "${train_dir}"

      # Universal
      # FINet
      elif [ "${model[i]}" == "finet" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --config "${config}" \
          --root "${run_dir}/train" \
          --project "${project}/${model[i]}" \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # HINet
      elif [ "${model[i]}" == "hinet" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --config "${config}" \
          --root "${run_dir}/train" \
          --project "${project}/${model[i]}" \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # UFormer
      elif [ "${model[i]}" == "uformer" ]; then
        cd "${universal_dir}/${model[i]}" || exit
        if [ "${task}" == "derain" ]; then
            echo -e "\nI have not prepared the training script for UFormer."
        fi
      fi
    done
  done
fi


## PREDICT
if [ "$run" == "predict" ]; then
  echo -e "\nPredicting"
  for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#variant[@]}; j++ )); do
      # Model initialization
      if [ "${variant[j]}" != "none" ] && [ "${variant[j]}" != "" ]; then
        model_variant="${model[i]}-${variant[j]}"
      else
        model_variant="${model[i]}"
      fi
      if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
        model_variant_weights="${model_variant}-${train[0]}-${suffix}"
        model_variant_suffix="${model_variant}-${suffix}"
      else
        model_variant_weights="${model_variant}-${train[0]}"
        model_variant_suffix="${model_variant}"
      fi
      config="${model[i]}_${train[0]/-/_}"
      # Weights initialization
      train_dir="${root_dir}/run/train/${project}/${model[i]}/${model_variant_weights}"
      train_weights_pt="${root_dir}/run/train/${project}/${model[i]}/${model_variant_weights}/weights/${checkpoint}.pt"
      train_weights_pth="${root_dir}/run/train/${project}/${model[i]}/${model_variant_weights}/weights/${checkpoint}.pth"
      train_weights_ckpt="${root_dir}/run/train/${project}/${model[i]}/${model_variant_weights}/weights/${checkpoint}.ckpt"
      zoo_weights_pt="${root_dir}/zoo/${project}/${model[i]}/${model_variant_weights}.pt"
      zoo_weights_pth="${root_dir}/zoo/${project}/${model[i]}/${model_variant_weights}.pth"
      zoo_weights_ckpt="${root_dir}/zoo/${project}/${model[i]}/${model_variant_weights}.ckpt"
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
      elif [ -f "$zoo_weights_ckpt" ]; then
        weights="${zoo_weights_ckpt}"
      elif [ -f "$zoo_weights_pth" ]; then
        weights="${zoo_weights_pth}"
      else
        weights="${zoo_weights_pt}"
      fi

      # LCDPNet
      if [ "${model[i]}" == "lcdpnet" ]; then
        cd "${llie_dir}/${model[i]}" || exit
        python -W ignore src/test.py \
          checkpoint_path="${root_dir}/zoo/${project}/${model[i]}/lcdpnet-ours.ckpt" \
          +image_size=512
      else
        for (( k=0; k<${#predict[@]}; k++ )); do
          if [ "${use_data_dir}" == "yes" ]; then
            output_dir="${root_dir}/data/${task}/predict/${model_variant_suffix}/${predict[k]}"
          else
            output_dir="${root_dir}/run/predict/${project}/${model_variant_suffix}/${predict[k]}"
          fi

          # De-Hazing
          # ZID
          if [ "${model[i]}" == "zid" ]; then
            cd "${dehaze_dir}/${model[i]}" || exit
            python -W ignore rw_dehazing.py \
              --input-dir "data/" \
              --output-dir "output/" \
              --num-iters 500
            # cd "${current_dir}" || exit
            # python -W ignore predict.py \
            #   --config "${model[i]}_jin2022" \
            #   --input-dir "${input_dirs[k]}" \
            #   --output-dir "${output_dir}" \
            #   --root "${output_dir}" \
            #   --project "${project}/${model[i]}" \
            #   --variant "${variant[j]}" \
            #   --weights "${weights}" \
            #   --image-size 512 \
            #   --devices "cuda:0" \
            #   --benchmark \
            #   --save-image \
            #   --verbose

          # De-Raining
          # IPT
          elif [ "${model[i]}" == "ipt" ]; then
            cd "${derain_dir}/${model[i]}" || exit
            python -W ignore main.py \
              --dir_data "${input_dirs[k]}" \
              --pretrain "${root_dir}/zoo/vision/enhance/derain/ipt/ipt-derain.pt" \
              --scale 1 \
              --derain  \
              --test_only \
              --save "${output_dir}" \
              --save_results \
              --save_models \
              --n_GPUs 2

          # LES
          # Jin2022
          elif [ "${model[i]}" == "jin2022" ]; then
            cd "${les_dir}/${model[i]}" || exit
            python -W ignore predict.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --data-name "${predict[k]}" \
              --phase "test" \
              --weights "${root_dir}/zoo/vision/enhance/les/jin2022/delighteffects_params_0600000.pt" \
              --image-size 512 \
              --benchmark

          # LLIE
          # EnlightenGAN
          elif [ "${model[i]}" == "enlightengan" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore infer/predict.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/enlightengan/enlightengan.onnx" \
              --image-size 512 \
              --benchmark
          # GCENet
          elif [ "${model[i]}" == "gcenet" ]; then
            cd "${current_dir}" || exit
            python -W ignore predict.py \
              --config "${config}" \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --root "${output_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --image-size 512 \
              --devices "cuda:0" \
              --benchmark \
              --save-image \
              --verbose
          # GCENetV2
          elif [ "${model[i]}" == "gcenetv2" ]; then
            cd "${current_dir}" || exit
            python -W ignore predict.py \
              --config "${config}" \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --root "${output_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --devices "cuda:0" \
              --benchmark \
              --save-image \
              --verbose
          # IAT
          elif [ "${model[i]}" == "iat" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore IAT_enhance/predict.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --exposure-weights "${root_dir}/zoo/vision/enhance/llie/iat/iat-exposure.pth" \
              --enhance-weights "${root_dir}/zoo/vision/enhance/llie/iat/iat-lol-v1.pth" \
              --image-size 512 \
              --normalize \
              --task "enhance" \
              --benchmark
          # KinD
          elif [ "${model[i]}" == "kind" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/kind" \
              --image-size 512 \
              --mode "test" \
              --benchmark
          # KinD++
          elif [ "${model[i]}" == "kind++" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/kind++" \
              --image-size 512 \
              --benchmark
          # LIME
          elif [ "${model[i]}" == "lime" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore demo.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --image-size 512 \
              --lime \
          # LLFlow
          elif [ "${model[i]}" == "llflow" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore code/test_unpaired_v2.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/llflow/llflow-lol-smallnet.pth" \
              --image-size 512 \
              --opt "code/confs/LOL_smallNet.yml" \
              --name "unpaired" \
              --benchmark
          # MBLLEN
          elif [ "${model[i]}" == "mbllen" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore main/test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --image-size 512 \
              --benchmark
          # PIE
          elif [ "${model[i]}" == "pie" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python main.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --image-size 512
          # RetinexDIP
          elif [ "${model[i]}" == "retinexdip" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore retinexdip.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/retinexdip/retinexdip-lol.pt" \
              --image-size 512 \
              --benchmark
          # RetinexNet
          elif [ "${model[i]}" == "retinexnet" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore predict.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/retinexnet" \
              --image-size 512 \
              --benchmark
          # RUAS
          elif [ "${model[i]}" == "ruas" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/ruas//ruas-lol.pt" \
              --image-size 512 \
              --gpu 0 \
              --seed 2 \
              --benchmark
          # SCI
          elif [ "${model[i]}" == "sci" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/sci/sci-medium.pt" \
              --image-size 512 \
              --gpu 0 \
              --seed 2 \
              --benchmark
          # SGZ
          elif [ "${model[i]}" == "sgz" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/sgz/sgz-lol.pt" \
              --image-size 512 \
              --benchmark
          # SNR-Aware
          elif [ "${model[i]}" == "snr" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore predict.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/snr/snr-lolv1.pth" \
              --opt "./options/test/LOLv1.yml" \
              --image-size 512 \
              --benchmark
          # StableLLVE
          elif [ "${model[i]}" == "stablellve" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/stablellve/stablellve-checkpoint.pth" \
              --image-size 512 \
              --benchmark
          # URetinex-Net
          elif [ "${model[i]}" == "uretinexnet" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --decom-model-low-weights "${root_dir}/zoo/vision/enhance/llie/uretinexnet/uretinexnet-init_low.pth" \
              --unfolding-model-weights "${root_dir}/zoo/vision/enhance/llie/uretinexnet/uretinexnet-unfolding.pth" \
              --adjust-model-weights "${root_dir}/zoo/vision/enhance/llie/uretinexnet/uretinexnet-L_adjust.pth" \
              --image-size 512 \
              --ratio 5 \
              --benchmark
          # UTVNet
          elif [ "${model[i]}" == "utvnet" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/utvnet/utvnet-model_test.pt" \
              --image-size 512 \
              --benchmark
          # Zero-ADCE
          elif [ "${model[i]}" == "zeroadce" ]; then
            cd "${current_dir}" || exit
            python -W ignore predict.py \
              --config "${config}" \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --root "${output_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --image-size 512 \
              --devices "cuda:0" \
              --benchmark \
              --save-image \
              --verbose
          # Zero-DCE
          elif [ "${model[i]}" == "zerodce" ]; then
            model_dir="${llie_dir}/${model[i]}"
            cd "${model_dir}" || exit
            python -W ignore lowlight_test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/zerodce/best.pth" \
              --image-size 512 \
              --benchmark
          # Zero-DCE++
          elif [ "${model[i]}" == "zerodce++" ]; then
            cd "${llie_dir}/${model[i]}" || exit
            python -W ignore lowlight_test.py \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --weights "${root_dir}/zoo/vision/enhance/llie/zerodce++/best.pth" \
              --image-size 512 \
              --benchmark

          # Universal
          # FINet
          elif [ "${model[i]}" == "finet" ]; then
            cd "${current_dir}" || exit
            python -W ignore predict.py \
              --config "${config}" \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --root "${output_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --image-size 256 \
              --resize \
              --devices "cuda:0" \
              --benchmark \
              --save-image \
              --verbose
          # HINet
          elif [ "${model[i]}" == "hinet" ]; then
            cd "${current_dir}" || exit
            python -W ignore predict.py \
              --config "${config}" \
              --input-dir "${input_dirs[k]}" \
              --output-dir "${output_dir}" \
              --root "${output_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --image-size 256 \
              --resize \
              --devices "cuda:0" \
              --benchmark \
              --save-image \
              --verbose
          # UFormer
          elif [ "${model[i]}" == "uformer" ]; then
            cd "${universal_dir}/${model[i]}" || exit
            if [ "${run}" == "derain" ]; then
              python -W ignore test/test_derain.py \
                --input-dir "${input_dirs[k]}" \
                --output-dir "${output_dir}" \
                --weights "${weights}" \
                --image-size 512
            fi
          fi
        done
      fi
    done
  done
fi


## EVALUATE
if [ "$run" == "evaluate" ]; then
  echo -e "\nEvaluate"
  cd "${current_dir}" || exit
  for (( k=0; k<${#predict[@]}; k++ )); do
    echo -e "\n${predict[k]}"

    for (( i=0; i<${#model[@]}; i++ )); do
      for (( j=0; j<${#variant[@]}; j++ )); do
          # Model initialization
          if [ "${variant[j]}" != "none" ]; then
            model_variant="${model[i]}-${variant[j]}"
          else
            model_variant="${model[i]}"
          fi
          if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
            model_variant_suffix="${model_variant}-${suffix}"
          else
            model_variant_suffix="${model_variant}"
          fi

          if [ "${predict[k]}" == "darkcityscapes" ]; then
            if [ "${use_data_dir}" == "yes" ]; then
              output_dir="${data_dir}/${task}/predict/${model_variant_suffix}/${predict[k]}/enhance"
            else
              output_dir="${root_dir}/run/predict/${project}/${model_variant_suffix}/${predict[k]}/enhance"
            fi
          else
            if [ "${use_data_dir}" == "yes" ]; then
              output_dir="${data_dir}/${task}/predict/${model_variant_suffix}/${predict[k]}"
            else
              output_dir="${root_dir}/run/predict/${project}/${model_variant_suffix}/${predict[k]}"
            fi
          fi

          if [ "${j}" == 0 ]; then
            python -W ignore metric.py \
              --input-dir "${output_dir}" \
              --target-dir "${root_dir}/data/llie/test/${predict[k]}/high" \
              --result-file "${current_dir}" \
              --name "${model_variant_suffix}" \
              --image-size 256 \
              --test-y-channel \
              --backend "piqa" \
              --backend "pyiqa" \
              --show-results \
              --metric "psnr" \
              --metric "psnry" \
              --metric "ssim" \
              --metric "ms-ssim" \
              --metric "lpips" \
              --metric "brisque" \
              --metric "niqe" \
              --metric "pi"
              # --name "${model[i]}" \
              # --variant "${variant[j]}" \
          else
            python -W ignore metric.py \
              --input-dir "${output_dir}" \
              --target-dir "${root_dir}/data/llie/test/${predict[k]}/high" \
              --result-file "${current_dir}" \
              --name "${model_variant_suffix}" \
              --image-size 256 \
              --resize \
              --test-y-channel \
              --backend "piqa" \
              --backend "pyiqa" \
              --append-results \
              --show-results \
              --metric "psnr" \
              --metric "psnry" \
              --metric "ssim" \
              --metric "ms-ssim" \
              --metric "lpips" \
              --metric "brisque" \
              --metric "niqe" \
              --metric "pi"
              # --name "${model[i]}" \
              # --variant "${variant[j]}" \
          fi
        done
    done
  done
fi


# PLOT
if [ "$run" == "plot" ]; then
  echo -e "\\nPlot"
  cd "${current_dir}" || exit
  if [ "${use_data_dir}" == "yes" ]; then
    input_dir="${data_dir}/${task}/predict"
    output_dir="${data_dir}/${task}/compare"
  else
    input_dir="${root_dir}/run/predict/${project}"
    output_dir="${root_dir}/run/predict/${project}/compare"
  fi
  python -W ignore plot.py \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --image-size 512 \
    --num-cols 8
fi


# DONE
cd "${current_dir}" || exit
