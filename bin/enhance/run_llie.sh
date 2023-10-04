#!/bin/bash
echo "$HOSTNAME"


# Fast Commands
#


# Constants
models=(
  "bimef"
  "bpdhe"
  "cvc"
  "deepupe"
  "dheci"
  "dong"
  "drbn"
  "eemefn"
  "enlightengan"       # https://github.com/arsenyinfo/EnlightenGAN-inference
  "excnet"
  "he"
  "iat"                # https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance
  "jed"
  "kind"               # https://github.com/zhangyhuaee/KinD
  "kind++"             # https://github.com/zhangyhuaee/KinD_plus
  "lcdpnet"            # https://github.com/onpix/LCDPNet
  "ldr"
  "lightennet"
  "lime"               # https://github.com/pvnieo/Low-light-Image-Enhancement
  "llflow"             # https://github.com/wyf0912/LLFlow
  "llnet"
  "mbllen"             # https://github.com/Lvfeifan/MBLLEN
  "mf"
  "multiscaleretinex"
  "npe"
  "pie"                # https://github.com/DavidQiuChao/PIE
  "retinexdip"         # https://github.com/zhaozunjin/RetinexDIP
  "retinexnet"         # https://github.com/weichen582/RetinexNet
  "rrm"                
  "ruas"               # https://github.com/KarelZhang/RUAS
  "sci"                # https://github.com/vis-opt-group/SCI
  "sdsd"               
  "sgz"                #
  "sice"
  "sid"
  "snr"                # https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
  "srie"
  "stablellve"         # https://github.com/zkawfanx/StableLLVE
  "uretinexnet"        # https://github.com/AndersonYong/URetinex-Net
  "utvnet"             # https://github.com/CharlieZCJ/UTVNet
  "wahe"
  "zerodce"            #
  "zerodce++"          #
  "zerodace"           #
)
train_datasets=(
  "gladnet"
  "llie"
  "lol"
  "sice"
  "sice-grad"
  "sice-mix"
  "sice-zerodce"
  "ve-lol"
  "ve-lol-syn"
)
predict_datasets=(
  # "darkface"
  # "deepupe"
  "dicm"
  # "exdark"
  "fusion"
  "lime"
  "lol"
  "mef"
  "npe"
  # "sice"
  "ve-lol"
  "ve-lol-syn"
  "vv"
)


# Input
machine=$HOSTNAME
model=${1:-"all"}
variant=${2:-"none"}
task=${3:-"predict"}
train_data=${4:-"lol"}
predict_data=${5:-"all"}
project=${6:-"vision/enhance/llie"}
use_data_dir=${7:-"yes"}

read -e -i "$model"        -p "Model [all ${models[*]}]: " model
read -e -i "$variant"      -p "Variant: " variant
read -e -i "$task"         -p "Task [train, evaluate, predict]: " task
read -e -i "$train_data"   -p "Train data [all ${train_datasets[*]}]: " train_data
read -e -i "$predict_data" -p "Predict data [all ${predict_datasets[*]}]: " predict_data
read -e -i "$project"      -p "Project: " project
read -e -i "$use_data_dir" -p "Use data_dir [yes, no]: " use_data_dir

echo -e "\n"
machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
model=$(echo $model | tr '[:upper:]' '[:lower:]')
model=($(echo $model | tr ',' '\n'))
# echo "${model[*]}"
variant=$(echo $variant | tr '[:upper:]' '[:lower:]')
variant=($(echo "$variant" | tr ',' '\n'))
echo "${variant[*]}"
task=$(echo $task | tr '[:upper:]' '[:lower:]')
train_data=$(echo $train_data | tr '[:upper:]' '[:lower:]')
train_data=($(echo $train_data | tr ',' '\n'))
# echo "${train_data[*]}"
predict_data=$(echo $predict_data | tr '[:upper:]' '[:lower:]')
predict_data=($(echo $predict_data | tr ',' '\n'))
# echo "${predict_data[*]}"
project=$(echo $project | tr '[:upper:]' '[:lower:]')


# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")

if [ "$model" == "all" ]; then
  declare -a model=()
  for i in ${!models[@]}; do
    model[i]="${models[i]}"
  done
else
  declare -a model=($model)
fi

if [ "$train_data" == "all" ]; then
  declare -a train_data=()
  for i in ${!train_datasets[@]}; do
    train_data[i]="${train_datasets[i]}"
  done
else
  declare -a train_data=($train_data)
fi

if [ "$predict_data" == "all" ]; then
  declare -a predict_data=()
  for i in ${!predict_datasets[@]}; do
    predict_data[i]="${predict_datasets[i]}"
  done
else
  declare -a predict_data=($predict_data)
fi

declare -a low_data_dirs=()
declare -a high_data_dirs=()
if [ "$task" == "train" ]; then
  for d in "${train_data[@]}"; do
    if [ "$d" == "gladnet" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/gladnet/low")
      high_data_dirs+=("${root_dir}/data/llie/train/gladnet/high")
    fi
    if [ "$d" == "lol" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/lol/low")
      high_data_dirs+=("${root_dir}/data/llie/train/lol/high")
    fi
    if [ "$d" == "sice" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/sice-part1/low")
      high_data_dirs+=("${root_dir}/data/llie/train/sice-part1/high")
    fi
    if [ "$d" == "sice-grad" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/sice-grad/low")
      high_data_dirs+=("${root_dir}/data/llie/train/sice-grad/high")
    fi
    if [ "$d" == "sice-mix" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/sice-mix/low")
      high_data_dirs+=("${root_dir}/data/llie/train/sice-mix/high")
    fi
    if [ "$d" == "sice-zerodce" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/sice-zerodce/low")
      high_data_dirs+=("${root_dir}/data/llie/train/sice-zerodce/high")
    fi
    if [ "$d" == "ve-lol" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/ve-lol/low")
      high_data_dirs+=("${root_dir}/data/llie/train/ve-lol/high")
    fi
    if [ "$d" == "ve-lol-sync" ]; then
      low_data_dirs+=("${root_dir}/data/llie/train/ve-lol-sync/low")
      high_data_dirs+=("${root_dir}/data/llie/train/ve-lol-sync/high")
    fi
  done
elif [ "$task" == "predict" ]; then
  for d in "${predict_data[@]}"; do
    if [ "$d" == "darkface" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/darkface/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "deepupe" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/deepupe/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "dicm" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/dicm/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "exdark" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/exdark/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "fusion" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/fusion/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "lime" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/lime/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "lol" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/lol/low")
      high_data_dirs+=("${root_dir}/data/llie/test/lol/high")
    fi
    if [ "$d" == "mef" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/mef/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "npe" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/npe/low")
      high_data_dirs+=("")
    fi
    if [ "$d" == "sice" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/sice-part2/low")
      high_data_dirs+=("${root_dir}/data/llie/test/sice-part2/high")
    fi
    if [ "$d" == "ve-lol" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/ve-lol/low")
      high_data_dirs+=("${root_dir}/data/llie/test/ve-lol/high")
    fi
    if [ "$d" == "ve-lol-syn" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/ve-lol-syn/low")
      high_data_dirs+=("${root_dir}/data/llie/test/ve-lol-syn/high")
    fi
    if [ "$d" == "vv" ]; then
      low_data_dirs+=("${root_dir}/data/llie/test/vv/low")
      high_data_dirs+=("")
    fi
  done
fi


# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  for (( i=0; i<${#model[@]}; i++ )); do
    # Model initialization
    if [ "${model[i]}" == "zerodace" ]; then
      model_dir="${current_dir}"
    else
      model_dir="${root_dir}/src/lib/${project}/${model[i]}"
    fi
    cd "${model_dir}" || exit

    if [ "$variant" != "none" ]; then
      model_variant="${model[i]}-${variant}"
    else
      model_variant="${model[i]}"
    fi
    train_dir="${root_dir}/run/train/${project}/${model_variant}-${train_data[0]}"
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
        num_epoch=100 \
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
        --data "${low_data_dirs[j]}" \
        --weights "${weights}" \
        --image-size 512 \
        --output-dir "${train_dir}"
    # RetinexDIP
    elif [ "${model[i]}" == "retinexnet" ]; then
      echo -e "\nI have not prepared the training script for RetinexNet."
    # RUAS
    elif [ "${model[i]}" == "ruas" ]; then
      python -W ignore train.py \
        --data "${low_data_dirs[j]}" \
        --weights "${weights}" \
        --load-pretrain false \
        --epoch 100 \
        --batch-size 1 \
        --report-freq 50 \
        --gpu 0 \
        --seed 2 \
        --checkpoints-dir "${train_dir}"
    # SCI
    elif [ "${model[i]}" == "sci" ]; then
      python -W ignore train.py \
        --data "${low_data_dirs[j]}" \
        --weights "${weights}" \
        --load-pretrain false \
        --batch-size 1 \
        --epochs 100 \
        --lr 0.0003 \
        --stage 3 \
        --cuda true \
        --gpu 0 \
        --seed 2 \
        --checkpoints-dir "${train_dir}"
    # SGZ
    elif [ "${model[i]}" == "sgz" ]; then
      python -W ignore train.py \
        --data "${low_data_dirs[j]}" \
        --weights "${weights}" \
        --load-pretrain false \
        --image-size 512 \
        --lr 0.0001 \
        --weight-decay 0.0001 \
        --grad-clip-norm 0.1 \
        --epochs 100 \
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
    # Zero-DCE
    elif [ "${model[i]}" == "zerodce" ]; then
      python -W ignore lowlight_train.py \
        --data "${low_data_dirs[j]}" \
        --weights "${weights}" \
        --load-pretrain false \
        --lr 0.0001 \
        --weight-decay 0.0001 \
        --grad-clip-norm 0.1 \
        --epochs 100 \
        --train-batch-size 8 \
        --val-batch-size 4 \
        --num-workers 4 \
        --display-iter 10 \
        --checkpoints-iter 10 \
        --checkpoints-dir "${train_dir}"
    # Zero-DCE++
    elif [ "${model[i]}" == "zerodce++" ]; then
      python -W ignore lowlight_train.py \
        --data "${low_data_dirs[j]}" \
        --weights "${weights}" \
        --load-pretrain false \
        --lr 0.0001 \
        --weight-decay 0.0001 \
        --grad-clip-norm 0.1 \
        --scale-factor 1 \
        --epochs 100 \
        --train-batch-size 8 \
        --val-batch-size 4 \
        --num-workers 4 \
        --display-iter 10 \
        --checkpoints-iter 10 \
        --checkpoints-dir "${train_dir}"
    # Zero-DACE
    elif [ "${model[i]}" == "zerodace" ]; then
      if [ "${variant[i]}" == "all" ]; then
        variants=(
          "0000"
          "0100" "0101" "0102" "0103" "0104" "0105" "0106"
          "0200" "0201" "0202" "0203" "0204" "0205" "0206" "0207" "0208" "0209" "0210" "0211" "0212"
          "0300" "0301" "0302" "0303" "0304" "0305"
          "0400" "0401" "0402" "0403" "0404"
          "0500"
          "0600"
        )
        for (( v=0; v<${#variants[@]}; v++ )); do
          model_variant="${model[i]}-${variants[v]}"
          train_dir="${root_dir}/run/train/${project}/${model_variant}-${train_data[0]}"
          python -W ignore train.py \
            --name "${model_variant}-${train_data[0]}" \
            --variant "${variants[v]}"
        done
      else
        python -W ignore train.py \
          --name "${model_variant}-${train_data[0]}" \
          --variant "${variant}"
      fi
    fi
  done
fi


# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  for (( i=0; i<${#model[@]}; i++ )); do
    # Model initialization
    if [ "${model[i]}" == "zerodace" ]; then
      model_dir="${current_dir}"
    else
      model_dir="${root_dir}/src/lib/${project}/${model[i]}"
    fi
    cd "${model_dir}" || exit

    if [ "$variant" != "none" ]; then
      model_variant="${model[i]}-${variant}"
    else
      model_variant="${model[i]}"
    fi
    train_dir="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}"
    train_weights_pt="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}/best.pt"
    train_weights_pth="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}/best.pth"
    train_weights_ckpt="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}/best.ckpt"
    zoo_weights_pt="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train_data[0]}.pt"
    zoo_weights_pth="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train_data[0]}.pth"
    zoo_weights_ckpt="${root_dir}/zoo/${project}/${model[i]}/${model_variant}-${train_data[0]}.ckpt"
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
        checkpoint_path="${root_dir}/zoo/${project}/${model[i]}/lcdpnet-ours.ckpt"  \
        +image_size=512
    else
      for (( j=0; j<${#predict_data[@]}; j++ )); do
        if [ "${use_data_dir}" == "yes" ]; then
          predict_dir="${root_dir}/data/llie/predict/${model_variant}/${predict_data[j]}"
        else
          predict_dir="${root_dir}/run/predict/${project}/${model_variant}/${predict_data[j]}"
        fi

        # EnlightenGAN
        if [ "${model[i]}" == "enlightengan" ]; then
          python -W ignore infer/predict.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # IAT
        elif [ "${model[i]}" == "iat" ]; then
          python -W ignore IAT_enhance/predict.py \
            --data "${low_data_dirs[j]}" \
            --exposure-weights "${root_dir}/zoo/${project}/${model[i]}/iat-exposure.pth" \
            --enhance-weights "${root_dir}/zoo/${project}/${model[i]}/iat-lol.pth" \
            --image-size 512 \
            --normalize \
            --task "enhance" \
            --output-dir "${predict_dir}"
        # KinD
        elif [ "${model[i]}" == "kind" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # KinD++
        elif [ "${model[i]}" == "kind++" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # LLFlow
        elif [ "${model[i]}" == "llflow" ]; then
          python -W ignore code/test_unpaired_v2.py \
            --data "${low_data_dirs[j]}" \
            --weights "${root_dir}/zoo/${project}/${model[i]}/llflow-lol-smallnet.pth" \
            --image-size 512 \
            --output-dir "${predict_dir}" \
            --opt "code/confs/LOL_smallNet.yml" \
            --name "unpaired"
        # LIME
        elif [ "${model[i]}" == "lime" ]; then
          python -W ignore demo.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --lime \
            --output-dir "${predict_dir}" \
        # MBLLEN
        elif [ "${model[i]}" == "mbllen" ]; then
          python -W ignore main/test.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --output-dir "${predict_dir}" \
        # PIE
        elif [ "${model[i]}" == "pie" ]; then
          python main.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --output-dir "${predict_dir}" \
        # RetinexDIP
        elif [ "${model[i]}" == "retinexdip" ]; then
          python -W ignore retinexdip.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # RetinexDIP
        elif [ "${model[i]}" == "retinexnet" ]; then
          python -W ignore main.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --phase "test" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # RUAS
        elif [ "${model[i]}" == "ruas" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --image-size 512 \
            --gpu 0 \
            --seed 2 \
            --output-dir "${predict_dir}"
        # SCI
        elif [ "${model[i]}" == "sci" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --image-size 512 \
            --gpu 0 \
            --seed 2 \
            --output-dir "${predict_dir}"
        # SGZ
        elif [ "${model[i]}" == "sgz" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # SNR-Aware
        elif [ "${model[i]}" == "snr" ]; then
          python -W ignore predict.py \
            --data "${low_data_dirs[j]}" \
            --weights "${root_dir}/zoo/${project}/${model[i]}/snr-lolv1.pth" \
            --opt "./options/test/LOLv1.yml" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # StableLLVE
        elif [ "${model[i]}" == "stablellve" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --weights "./checkpoint.pth" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # URetinex-Net
        elif [ "${model[i]}" == "uretinexnet" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --decom-model-low-weights "${root_dir}/zoo/${project}/${model[i]}/${train_data[0]}/init_low-lol.pth" \
            --unfolding-model-weights "${root_dir}/zoo/${project}/${model[i]}/${train_data[0]}/unfolding-lol.pth" \
            --adjust-model-weights "${root_dir}/zoo/${project}${model[i]}/${train_data[0]}/L_adjust-lol.pth" \
            --image-size 512 \
            --ratio 5 \
            --output-dir "${predict_dir}"
        # UTVNet
        elif [ "${model[i]}" == "utvnet" ]; then
          python -W ignore test.py \
            --data "${low_data_dirs[j]}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # Zero-DCE
        elif [ "${model[i]}" == "zerodce" ]; then
          python -W ignore lowlight_test.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # Zero-DCE++
        elif [ "${model[i]}" == "zerodce++" ]; then
          python -W ignore lowlight_test.py \
            --data "${low_data_dirs[j]}" \
            --weights "${weights}" \
            --image-size 512 \
            --output-dir "${predict_dir}"
        # Zero-DACE
        elif [ "${model[i]}" == "zerodace" ]; then
          if [ "${variant[i]}" == "all" ]; then
            variants=(
              "0000"
              "0100" "0101" "0102" "0103" "0104" "0105" "0106"
              "0200" "0201" "0202" "0203" "0204" "0205" "0206" "0207" "0208" "0209" "0210" "0211" "0212"
              "0300" "0301" "0302" "0303" "0304" "0305"
              "0400" "0401" "0402" "0403" "0404"
              "0500"
              "0600"
            )
            for (( v=0; v<${#variants[@]}; v++ )); do
              model_variant="${model[i]}-${variants[v]}"
              weights="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}/weights/best.pt"
              python -W ignore predict.py \
                --data "${low_data_dirs[j]}" \
                --config "${model[i]}_sice_zerodce" \
                --root "${predict_dir}" \
                --project "${project}/${model[i]}" \
                --variant "${variants[v]}" \
                --weights "${weights}" \
                --num_iters 6 \
                --unsharp_sigma 1.5 \
                --image-size 512 \
                --output-dir "${predict_dir}"
            done
          else
            # python -W ignore train.py \
            #   --name "${model_variant}-${train_data[0]}" \
            #   --variant "${variant}"
            weights="${root_dir}/run/train/${project}/${model[i]}/${model_variant}-${train_data[0]}/weights/best.pt"
            python -W ignore predict.py \
              --data "${low_data_dirs[j]}" \
              --config "${model[i]}_llie" \
              --root "${predict_dir}" \
              --project "${project}/${model[i]}" \
              --variant "${variant[0]}" \
              --weights "${weights}" \
              --num_iters 6 \
              --unsharp_sigma 1.5 \
              --image-size 512 \
              --output-dir "${predict_dir}"
          fi
        fi
      done
    fi
  done
fi


# Evaluate
if [ "$task" == "evaluate" ]; then
  echo -e "\nEvaluate"
  cd "${current_dir}" || exit
  for (( i=0; i<${#model[@]}; i++ )); do
    for (( j=0; j<${#predict_data[@]}; j++ )); do
        if [ "$variant" != "none" ]; then
          model_variant="${model[i]}-${variant}"
        else
          model_variant="${model[i]}"
        fi
        if [ "${use_data_dir}" == "yes" ]; then
          predict_dir="${root_dir}/data/llie/predict/${model_variant}/${predict_data[j]}"
        else
          predict_dir="${root_dir}/run/predict/${project}/${model_variant}/${predict_data[j]}"
        fi
        python -W ignore metric.py \
          --image-dir "${predict_dir}" \
          --target-dir "${root_dir}/data/llie/test/${predict_data[j]}/high" \
          --result-file "${current_dir}" \
          --model-name "${model[i]}" \
          --image-size 512 \
          --resize \
          --test-y-channel \
          --backend "piqa" \
          --append-results \
          --metric "brisque" \
          --metric "niqe" \
          --metric "pi" \
          --metric "fsim" \
          --metric "haarpsi" \
          --metric "lpips" \
          --metric "mdsi" \
          --metric "ms-gmsd" \
          --metric "ms-ssim" \
          --metric "psnr" \
          --metric "ssim" \
          --metric "vsi"
      done
  done
fi


# Done
cd "${root_dir}" || exit
