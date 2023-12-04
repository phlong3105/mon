#!/bin/bash
echo "$HOSTNAME"


# Fast Commands
# ./run_les.sh zerodcev2 none train   sice-zerodce lol vision/enhance/llie no last
# ./run_les.sh zerodcev2 none predict sice-zerodce all vision/enhance/llie no last


# Constants
models=(
  "enlightengan"       # https://github.com/arsenyinfo/EnlightenGAN-inference
  "gcenet"             # Our model
  "gcenetv2"           # Our model
  "iat"                # https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance
  "kind"               # https://github.com/zhangyhuaee/KinD
  "kind++"             # https://github.com/zhangyhuaee/KinD_plus
  "lcdpnet"            # https://github.com/onpix/LCDPNet
  "lime"               # https://github.com/pvnieo/Low-light-Image-Enhancement
  "llflow"             # https://github.com/wyf0912/LLFlow
  "mbllen"             # https://github.com/Lvfeifan/MBLLEN
  "pie"                # https://github.com/DavidQiuChao/PIE
  "retinexdip"         # https://github.com/zhaozunjin/RetinexDIP
  "retinexnet"         # https://github.com/weichen582/RetinexNet
  "ruas"               # https://github.com/KarelZhang/RUAS
  "sci"                # https://github.com/vis-opt-group/SCI
  "sgz"                #
  "snr"                # https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
  "stablellve"         # https://github.com/zkawfanx/StableLLVE
  "uretinexnet"        # https://github.com/AndersonYong/URetinex-Net
  "utvnet"             # https://github.com/CharlieZCJ/UTVNet
  "zeroadce"           #
  "zerodce"            #
  "zerodce++"          #
)
train_datasets=(
  "jin2022"
  "ledlight"
  "sice-zerodce"
  "syn-light-effects"
)
predict_datasets=(
  "jin2022"
  "ledlight"
  "syn-light-effects"
)


# Input
machine=$HOSTNAME
model=${1:-"all"}
variant=${2:-"none"}
suffix=${3:-"none"}
task=${4:-"predict"}
epochs=${5:-100}
train_data=${6:-"lol"}
predict_data=${7:-"all"}
project=${8:-"vision/enhance/llie"}
use_data_dir=${9:-"yes"}
checkpoint=${10:-"best"}

read -e -i "$model"        -p "Model [all ${models[*]}]: " model
read -e -i "$variant"      -p "Variant: " variant
read -e -i "$suffix"       -p "Suffix: " suffix
read -e -i "$task"         -p "Task [train, evaluate, predict, plot]: " task
read -e -i "$epochs"       -p "Epochs: " epochs
read -e -i "$train_data"   -p "Train data [all ${train_datasets[*]}]: " train_data
read -e -i "$predict_data" -p "Predict data [all ${predict_datasets[*]}]: " predict_data
read -e -i "$project"      -p "Project: " project
read -e -i "$use_data_dir" -p "Use data_dir [yes, no]: " use_data_dir
read -e -i "$checkpoint"   -p "Checkpoint type [best, last]: " checkpoint

echo -e "\n"
machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
model=$(echo $model | tr '[:upper:]' '[:lower:]')
model=($(echo $model | tr ',' '\n'))

variant=$(echo $variant | tr '[:upper:]' '[:lower:]')
variant=($(echo "$variant" | tr ',' '\n'))

task=$(echo $task | tr '[:upper:]' '[:lower:]')

train_data=$(echo $train_data | tr '[:upper:]' '[:lower:]')
train_data=($(echo $train_data | tr ',' '\n'))

predict_data=$(echo $predict_data | tr '[:upper:]' '[:lower:]')
predict_data=($(echo $predict_data | tr ',' '\n'))

project=$(echo $project | tr '[:upper:]' '[:lower:]')


# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")

if [ "${model[0]}" == "all" ]; then
  declare -a model=()
  for i in ${!models[@]}; do
    model[i]="${models[i]}"
  done
else
  declare -a model=("${model[0]}")
fi

if [ "${train_data[0]}" == "all" ]; then
  declare -a train_data=()
  for i in ${!train_datasets[@]}; do
    train_data[i]="${train_datasets[i]}"
  done
else
  declare -a train_data=("${train_data[0]}")
fi

if [ "${predict_data[0]}" == "all" ]; then
  declare -a predict_data=()
  for i in ${!predict_datasets[@]}; do
    predict_data[i]="${predict_datasets[i]}"
  done
fi

declare -a light_effects_data_dirs=()
declare -a clear_data_dirs=()
if [ "$task" == "train" ]; then
  for d in "${train_data[@]}"; do
    if [ "$d" == "jin2022" ]; then
      light_effects_data_dirs+=("${root_dir}/data/les/test/jin2022/light-effects")
      light_effects_data_dirs+=("${root_dir}/data/les/test/jin2022/clear")
      clear_data_dirs+=("")
    fi
    if [ "$d" == "ledlight" ]; then
      light_effects_data_dirs+=("${root_dir}/data/les/test/ledlight/light-effects")
      clear_data_dirs+=("${root_dir}/data/les/test/ledlight/clear")
    fi
    if [ "$d" == "syn-light-effects" ]; then
      light_effects_data_dirs+=("${root_dir}/data/les/test/syn-light-effects/light-effects")
      clear_data_dirs+=("${root_dir}/data/les/test/syn-light-effects/clear")
    fi
  done
elif [ "$task" == "predict" ]; then
  for d in "${predict_data[@]}"; do
    if [ "$d" == "jin2022" ]; then
      light_effects_data_dirs+=("${root_dir}/data/les/test/jin2022/light-effects")
      light_effects_data_dirs+=("${root_dir}/data/les/test/jin2022/clear")
      clear_data_dirs+=("")
    fi
    if [ "$d" == "ledlight" ]; then
      light_effects_data_dirs+=("${root_dir}/data/les/test/ledlight/light-effects")
      clear_data_dirs+=("${root_dir}/data/les/test/ledlight/clear")
    fi
    if [ "$d" == "sice-zerodce" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/sice-zerodce/low")
      high_data_dirs+=("${root_dir}/data/llie/train/sice-zerodce/high")
    fi
    if [ "$d" == "syn-light-effects" ]; then
      light_effects_data_dirs+=("${root_dir}/data/les/test/syn-light-effects/light-effects")
      clear_data_dirs+=("${root_dir}/data/les/test/syn-light-effects/clear")
    fi
  done
fi


# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#variant[@]}; j++ )); do
      # Model initialization
      if [ "${model[i]}" == "gcenet" ] || [ "${model[i]}" == "gcenetv2" ] || [ "${model[i]}" == "zeroadce" ]; then
        model_dir="${current_dir}"
      else
        model_dir="${root_dir}/src/lib/${project}/${model[i]}"
      fi
      cd "${model_dir}" || exit

      if [ "${variant[j]}" != "none" ] && [ "${variant[j]}" != "" ]; then
        model_variant="${model[i]}-${variant[j]}"
      else
        model_variant="${model[i]}"
      fi

      if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
        name="${model_variant}-${train_data[0]}-${suffix}"
      else
        name="${model_variant}-${train_data[0]}"
      fi

      train_dir="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}"
      zoo_weights_pt="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train_data[0]}.pt"
      zoo_weights_pth="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train_data[0]}.pth"
      zoo_weights_ckpt="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train_data[0]}.ckpt"
      if [ -f "$zoo_weights_ckpt" ]; then
        weights="${zoo_weights_ckpt}"
      elif [ -f "$zoo_weights_pth" ]; then
        weights="${zoo_weights_pth}"
      else
        weights="${zoo_weights_pt}"
      fi

      # EnlightenGAN
      if [ "${model[i]}" == "enlightengan" ]; then
        echo -e "\nI have not prepared the training script for EnlightenGAN."
      # GCENet
      elif [ "${model[i]}" == "gcenet" ]; then
        python -W ignore train.py \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # GCENetV2
      elif [ "${model[i]}" == "gcenetv2" ]; then
        python -W ignore train.py \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # IAT
      elif [ "${model[i]}" == "iat" ]; then
        echo -e "\nI have not prepared the training script for IAT."
      # KinD
      elif [ "${model[i]}" == "kind" ]; then
        echo -e "\nI have not prepared the training script for KinD."
      # KinD++
      elif [ "${model[i]}" == "kind++" ]; then
        echo -e "\nI have not prepared the training script for KinD++."
      # LCDPNet
      elif [ "${model[i]}" == "lcdpnet" ]; then
        python -W ignore src/train.py \
          name="lcdpnet-lol" \
          num_epoch="$epochs" \
          log_every=2000 \
          valid_every=20
      # LLFlow
      elif [ "${model[i]}" == "llflow" ]; then
        echo -e "\nI have not prepared the training script for LLFlow."
      # LIME
      elif [ "${model[i]}" == "lime" ]; then
        echo -e "\nLIME does not need any training."
      # MBLLEN
      elif [ "${model[i]}" == "mbllen" ]; then
        echo -e "\nI have not prepared the training script for MBLLEN."
      # PIE
      elif [ "${model[i]}" == "pie" ]; then
        echo -e "\nPIE does not need any training."
      # RetinexDIP
      elif [ "${model[i]}" == "retinexdip" ]; then
        echo -e "\nRetinexNet does not need any training."
        python -W ignore retinexdip.py \
          --data "${light_effects_data_dirs[j]}" \
          --weights "${weights}" \
          --image-size 512 \
          --output-dir "${train_dir}"
      # RetinexDIP
      elif [ "${model[i]}" == "retinexnet" ]; then
        echo -e "\nI have not prepared the training script for RetinexNet."
      # RUAS
      elif [ "${model[i]}" == "ruas" ]; then
        python -W ignore train.py \
          --data "${light_effects_data_dirs[j]}" \
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
        python -W ignore train.py \
          --data "${light_effects_data_dirs[j]}" \
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
        python -W ignore train.py \
          --data "${light_effects_data_dirs[j]}" \
          --weights "${weights}" \
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
        echo -e "\nI have not prepared the training script for SNR-Aware."
      # StableLLVE
      elif [ "${model[i]}" == "stablellve" ]; then
        echo -e "\nI have not prepared the training script for StableLLVE."
      # URetinex-Net
      elif [ "${model[i]}" == "uretinexnet" ]; then
        echo -e "\nI have not prepared the training script for URetinex-Net."
      # UTVNet
      elif [ "${model[i]}" == "utvnet" ]; then
        echo -e "\nI have not prepared the training script for UTVNet."
      # Zero-ADCE
      elif [ "${model[i]}" == "zeroadce" ]; then
        python -W ignore train.py \
          --name "${name}" \
          --variant "${variant[j]}" \
          --max-epochs "$epochs"
      # Zero-DCE
      elif [ "${model[i]}" == "zerodce" ]; then
        python -W ignore lowlight_train.py \
          --data "${light_effects_data_dirs[j]}" \
          --weights "${weights}" \
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
        python -W ignore lowlight_train.py \
          --data "${light_effects_data_dirs[j]}" \
          --weights "${weights}" \
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
      fi
    done
  done
fi


# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#variant[@]}; j++ )); do
      # Model initialization
      if [ "${model[i]}" == "gcenet" ] || [ "${model[i]}" == "gcenetv2" ] || [ "${model[i]}" == "zeroadce" ]; then
        model_dir="${current_dir}"
      else
        model_dir="${root_dir}/src/lib/${project}/${model[i]}"
      fi
      cd "${model_dir}" || exit
      
      if [ "${variant[j]}" != "none" ] && [ "${variant[j]}" != "" ]; then
        model_variant="${model[i]}-${variant[j]}"
      else
        model_variant="${model[i]}"
      fi

      if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
        model_variant_weights="${model_variant}-${train_data[0]}-${suffix}"
        model_variant_suffix="${model_variant}-${suffix}"
      else
        model_variant_weights="${model_variant}-${train_data[0]}"
        model_variant_suffix="${model_variant}"
      fi

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
        python -W ignore src/test.py \
          checkpoint_path="${root_dir}/zoo/${project}/${model[i]}/lcdpnet-ours.ckpt" \
          +image_size=512
      else
        for (( k=0; k<${#predict_data[@]}; k++ )); do
          if [ "${use_data_dir}" == "yes" ]; then
            predict_dir="${root_dir}/data/llie/predict/${model_variant_suffix}/${predict_data[k]}"
          else
            predict_dir="${root_dir}/run/predict/${project}/${model_variant_suffix}/${predict_data[k]}"
          fi

          # EnlightenGAN
          if [ "${model[i]}" == "enlightengan" ]; then
            python -W ignore infer/predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # GCENet
          elif [ "${model[i]}" == "gcenet" ]; then
            python -W ignore predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --config "${model[i]}_sice_zerodce" \
              --root "${predict_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --num_iters 8 \
              --image-size 512 \
              --save-image \
              --benchmark \
              --output-dir "${predict_dir}"
          # GCENetV2
          elif [ "${model[i]}" == "gcenetv2" ]; then
            python -W ignore predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --config "${model[i]}_sice_zerodce" \
              --root "${predict_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --num_iters 8 \
              --image-size 512 \
              --save-image \
              --output-dir "${predict_dir}"
          # IAT
          elif [ "${model[i]}" == "iat" ]; then
            python -W ignore IAT_enhance/predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --exposure-weights "${root_dir}/zoo/${project}/${model[i]}/iat-exposure.pth" \
              --enhance-weights "${root_dir}/zoo/${project}/${model[i]}/iat-lol-v1.pth" \
              --image-size 512 \
              --normalize \
              --task "enhance" \
              --output-dir "${predict_dir}"
          # KinD
          elif [ "${model[i]}" == "kind" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # KinD++
          elif [ "${model[i]}" == "kind++" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # LIME
          elif [ "${model[i]}" == "lime" ]; then
            python -W ignore demo.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --lime \
              --output-dir "${predict_dir}" \
          # LLFlow
          elif [ "${model[i]}" == "llflow" ]; then
            python -W ignore code/test_unpaired_v2.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "weights/llflow-lol-smallnet.pth" \
              --image-size 512 \
              --output-dir "${predict_dir}" \
              --opt "code/confs/LOL_smallNet.yml" \
              --name "unpaired"
          # MBLLEN
          elif [ "${model[i]}" == "mbllen" ]; then
            python -W ignore main/test.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --output-dir "${predict_dir}" \
          # PIE
          elif [ "${model[i]}" == "pie" ]; then
            python main.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --output-dir "${predict_dir}" \
          # RetinexDIP
          elif [ "${model[i]}" == "retinexdip" ]; then
            python -W ignore retinexdip.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${weights}" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # RetinexNet
          elif [ "${model[i]}" == "retinexnet" ]; then
            python -W ignore predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # RUAS
          elif [ "${model[i]}" == "ruas" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/ruas-lol.pt" \
              --image-size 512 \
              --gpu 0 \
              --seed 2 \
              --output-dir "${predict_dir}"
          # SCI
          elif [ "${model[i]}" == "sci" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/sci-medium.pt" \
              --image-size 512 \
              --gpu 0 \
              --seed 2 \
              --output-dir "${predict_dir}"
          # SGZ
          elif [ "${model[i]}" == "sgz" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/sgz-lol.pt" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # SNR-Aware
          elif [ "${model[i]}" == "snr" ]; then
            python -W ignore predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/snr-lolv1.pth" \
              --opt "./options/test/LOLv1.yml" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # StableLLVE
          elif [ "${model[i]}" == "stablellve" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/stablellve-checkpoint.pth" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # URetinex-Net
          elif [ "${model[i]}" == "uretinexnet" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --decom-model-low-weights "${root_dir}/zoo/${project}/${model[i]}/uretinexnet-init_low.pth" \
              --unfolding-model-weights "${root_dir}/zoo/${project}/${model[i]}/uretinexnet-unfolding.pth" \
              --adjust-model-weights "${root_dir}/zoo/${project}/${model[i]}/uretinexnet-L_adjust.pth" \
              --image-size 512 \
              --ratio 5 \
              --output-dir "${predict_dir}"
          # UTVNet
          elif [ "${model[i]}" == "utvnet" ]; then
            python -W ignore test.py \
              --data "${light_effects_data_dirs[k]}" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # Zero-ADCE
          elif [ "${model[i]}" == "zeroadce" ]; then
            python -W ignore predict.py \
              --data "${light_effects_data_dirs[k]}" \
              --config "${model[i]}_sice_zerodce" \
              --root "${predict_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[j]}" \
              --weights "${weights}" \
              --num_iters 8 \
              --image-size 512 \
              --save-image \
              --output-dir "${predict_dir}"
          # Zero-DCE
          elif [ "${model[i]}" == "zerodce" ]; then
            python -W ignore lowlight_test.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/best.pth" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          # Zero-DCE++
          elif [ "${model[i]}" == "zerodce++" ]; then
            python -W ignore lowlight_test.py \
              --data "${light_effects_data_dirs[k]}" \
              --weights "${root_dir}/zoo/${project}/${model[i]}/best.pth" \
              --image-size 512 \
              --output-dir "${predict_dir}"
          fi
        done
      fi
    done
  done
fi


# Evaluate
if [ "$task" == "evaluate" ]; then
  echo -e "\nEvaluate"
  cd "${current_dir}" || exit
  for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#variant[@]}; j++ )); do
      for (( k=0; k<${#predict_data[@]}; k++ )); do
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

          if [ "${predict_data[k]}" == "darkcityscapes" ]; then
            if [ "${use_data_dir}" == "yes" ]; then
              predict_dir="${root_dir}/data/llie/predict/${model_variant_suffix}/${predict_data[k]}/enhance"
            else
              predict_dir="${root_dir}/run/predict/${project}/${model_variant_suffix}/${predict_data[k]}/enhance"
            fi
          else
            if [ "${use_data_dir}" == "yes" ]; then
              predict_dir="${root_dir}/data/llie/predict/${model_variant_suffix}/${predict_data[k]}"
            else
              predict_dir="${root_dir}/run/predict/${project}/${model_variant_suffix}/${predict_data[k]}"
            fi
          fi

          python -W ignore metric.py \
            --image-dir "${predict_dir}" \
            --target-dir "${root_dir}/data/llie/test/${predict_data[k]}/high" \
            --result-file "${current_dir}" \
            --name "${model_variant_suffix}" \
            --image-size 256 \
            --resize \
            --test-y-channel \
            --backend "piqa" \
            --append-results \
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
            # --image-size 256 \

          python -W ignore metric.py \
            --image-dir "${predict_dir}" \
            --target-dir "${root_dir}/data/llie/test/${predict_data[k]}/high" \
            --result-file "${current_dir}" \
            --name "${model_variant_suffix}" \
            --image-size 256 \
            --resize \
            --test-y-channel \
            --backend "pyiqa" \
            --append-results \
            --metric "psnry" \
            --metric "brisque" \
            --metric "niqe" \
            --metric "pi"
            # --metric "psnr" \
            # --metric "psnry" \
            # --metric "ssim" \
            # --metric "ms-ssim" \
            # --metric "lpips" \
            # --metric "brisque" \
            # --metric "niqe" \
            # --metric "pi"
            # --name "${model[i]}" \
            # --variant "${variant[j]}" \
        done
    done
  done
fi


# Plot
if [ "$task" == "plot" ]; then
  echo -e "\\nPlot"
  cd "${current_dir}" || exit
  if [ "${use_data_dir}" == "yes" ]; then
    predict_dir="${root_dir}/data/llie/predict"
    output_dir="${root_dir}/data/llie/compare"
  else
    predict_dir="${root_dir}/run/predict/${project}"
    output_dir="${root_dir}/run/predict/${project}/compare"
  fi
  python -W ignore plot.py \
    --image-dir "${predict_dir}" \
    --image-size 512 \
    --num-cols 8 \
    --output-dir "${output_dir}"
fi


# Done
cd "${current_dir}" || exit
