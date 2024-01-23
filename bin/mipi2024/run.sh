#!/bin/bash
echo "$HOSTNAME"

## CONSTANTS
# Task
tasks=(
  "les"
)
# Models
les_models=(
  "jin2022"       # https://github.com/jinyeying/night-enhancement
  "uformer"       # https://github.com/ZhendongWang6/Uformer
)
# Datasets
les_datasets=(
  "flarereal800"
  "ledlight"
  "light-effect"
  "mipiflare"
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
project=${8:-"mipi2024/"}
epochs=${9:-100}
use_data_dir=${10:-"no"}
checkpoint=${11:-"best"}

# User Input
read -e -i "$task" -p "Task [${tasks[*]}]: " task
read -e -i "$run"  -p "Task [train, predict, evaluate, plot]: " run
task=$(echo $task | tr '[:upper:]' '[:lower:]')
run=$(echo $run | tr '[:upper:]' '[:lower:]')
if [ "$task" == "les" ]; then
  read -e -i "$train"   -p "Train data [${les_datasets[*]}]: " train
  read -e -i "$predict" -p "Predict data [${les_datasets[*]}]: " predict
  read -e -i "$model"   -p "Model [${les_models[*]}]: " model
else
  echo -e "\nWrong task"
  exit 1
fi
if [ "$project" == "vision/enhance/" ]; then
  project="${project}${task}"
fi
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
les_dir="${root_dir}/src/lib/vision/enhance/les"
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
  # LES
  if [ "$d" == "flarereal800" ]; then
    if [ ${split} == "train" ]; then
      input_dirs+=("${data_dir}/les/train/${d}/flare")
      target_dirs+=("${data_dir}/les/train/${d}/clear")
    else
      input_dirs+=("${data_dir}/les/val/${d}/flare")
      target_dirs+=("")
    fi
  fi
  if [ "$d" == "ledlight" ]; then
    input_dirs+=("${data_dir}/les/test/${d}/light-effects")
    target_dirs+=("${data_dir}/les/test/${d}/clear")
  fi
  if [ "$d" == "light-effect" ]; then
    input_dirs+=("${data_dir}/les/train/${d}/light-effects")
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
        if [[ ${variant[j]} == *"${model[i]}"* ]]; then
          model_variant="${variant[j]}"
        else
          model_variant="${model[i]}-${variant[j]}"
        fi
      else
        model_variant="${model[i]}"
      fi
      if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
        fullname="${model_variant}-${train[0]}-${suffix}"
      else
        fullname="${model_variant}-${train[0]}"
      fi
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

      # LES
      # Jin2022
      if [ "${model[i]}" == "jin2022" ]; then
        cd "${les_dir}/${model[i]}" || exit
        python -W ignore demo_all.py \
          --data "${input_dirs[k]}" \
          --image-size 512 \
          --output-dir "${train_dir}/visual" \
          --checkpoint-dir "${train_dir}"

      # Universal
      # UFormer
      elif [ "${model[i]}" == "uformer" ]; then
        cd "${current_dir}" || exit
        python -W ignore train.py \
          --name "${model[i]}" \
          --variant "${variant[j]}" \
          --data "${train[0]}" \
          --root "${run_dir}/train" \
          --project "${project}/${model[i]}" \
          --fullname "${fullname}" \
          --max-epochs "$epochs" \
          --verbose
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
        if [[ ${variant[j]} == *"${model[i]}"* ]]; then
          model_variant="${variant[j]}"
        else
          model_variant="${model[i]}-${variant[j]}"
        fi
      else
        model_variant="${model[i]}"
      fi
      if [ "$suffix" != "none" ] && [ "$suffix" != "" ]; then
        model_weights="${model_variant}-${train[0]}-${suffix}"
        fullname="${model_variant}-${suffix}"
      else
        model_weights="${model_variant}-${train[0]}"
        fullname="${model_variant}"
      fi
      # Weights initialization
      train_dir="${root_dir}/run/train/${project}/${model[i]}/${model_weights}"
      train_weights_pt="${root_dir}/run/train/${project}/${model[i]}/${model_weights}/weights/${checkpoint}.pt"
      train_weights_pth="${root_dir}/run/train/${project}/${model[i]}/${model_weights}/weights/${checkpoint}.pth"
      train_weights_ckpt="${root_dir}/run/train/${project}/${model[i]}/${model_weights}/weights/${checkpoint}.ckpt"
      zoo_weights_pt="${root_dir}/zoo/${project}/${model[i]}/${model_weights}.pt"
      zoo_weights_pth="${root_dir}/zoo/${project}/${model[i]}/${model_weights}.pth"
      zoo_weights_ckpt="${root_dir}/zoo/${project}/${model[i]}/${model_weights}.ckpt"
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

      for (( k=0; k<${#predict[@]}; k++ )); do
        if [ "${use_data_dir}" == "yes" ]; then
          output_dir="${root_dir}/data/${task}/predict/${fullname}/${predict[k]}"
        else
          output_dir="${root_dir}/run/predict/${project}/${fullname}/${predict[k]}"
        fi
      
        # LES
        # Jin2022
        if [ "${model[i]}" == "jin2022" ]; then
          cd "${les_dir}/${model[i]}" || exit
          python -W ignore predict.py \
            --input-dir "${input_dirs[k]}" \
            --output-dir "${output_dir}" \
            --data-name "${predict[k]}" \
            --phase "test" \
            --weights "${root_dir}/zoo/vision/enhance/les/jin2022/delighteffects_params_0600000.pt" \
            --image-size 512 \
            --benchmark
      
        # Universal
        # UFormer
        elif [ "${model[i]}" == "uformer" ]; then
          cd "${current_dir}" || exit
          python -W ignore predict.py \
            --input-dir "${input_dirs[k]}" \
            --output-dir "${output_dir}" \
            --name "${model[i]}" \
            --variant "${variant[j]}" \
            --data "${train[0]}" \
            --root "${output_dir}" \
            --project "${project}/${model[i]}" \
            --fullname "${fullname}" \
            --weights "${weights}" \
            --image-size 256 \
            --devices "cuda:0" \
            --benchmark \
            --save-image \
            --verbose
        fi
      done
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
            fullname="${model_variant}-${suffix}"
          else
            fullname="${model_variant}"
          fi

          if [ "${predict[k]}" == "darkcityscapes" ]; then
            if [ "${use_data_dir}" == "yes" ]; then
              output_dir="${data_dir}/${task}/predict/${fullname}/${predict[k]}/enhance"
            else
              output_dir="${root_dir}/run/predict/${project}/${fullname}/${predict[k]}/enhance"
            fi
          else
            if [ "${use_data_dir}" == "yes" ]; then
              output_dir="${data_dir}/${task}/predict/${fullname}/${predict[k]}"
            else
              output_dir="${root_dir}/run/predict/${project}/${fullname}/${predict[k]}"
            fi
          fi

          if [ "${j}" == 0 ]; then
            python -W ignore metric.py \
              --input-dir "${output_dir}" \
              --target-dir "${root_dir}/data/llie/test/${predict[k]}/high" \
              --result-file "${current_dir}" \
              --name "${fullname}" \
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
              --name "${fullname}" \
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
