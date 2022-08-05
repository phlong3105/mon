#!/bin/bash

# Install:
# chmod +x install.sh
# conda init bash
# ./install.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$(dirname "$current_dir")


# Add conda-forge channel
echo "Add 'conda-forge':"
conda config --append channels conda-forge

# Update 'base' env
echo "Update 'base' environment:"
conda update --a --y
pip install --upgrade pip


case "$OSTYPE" in
  linux*)
    echo "Linux / WSL"
    # Create `one` env
    env_yml_path="${current_dir}/environment.yml"
    if { conda env list | grep 'one'; } >/dev/null 2>&1; then
      echo "Update 'one' environment:"
      conda env update --name one -f "${env_yml_path}"
    else
      echo "Create 'one' environment:"
      conda env create -f "${env_yml_path}"
    fi
    eval "$(conda shell.bash hook)"
    conda activate one
    pip install --upgrade pip
    # Install `mish-cuda`
    # echo "Install 'mish-cuda':"
    # pip install git+https://github.com/JunnYu/mish-cuda.git
    # Remove `cv2/plugin` folder
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packes/cv2/plugin
    ;;
  darwin*)
    echo "MacOS"
    # Create `one` env
    env_yml_path="${current_dir}/environment_mac.yml"
    if { conda env list | grep 'one'; } >/dev/null 2>&1; then
      echo "Update 'one' environment:"
      conda env update --name one -f "${env_yml_path}"
    else
      echo "Create 'one' environment:"
      conda env create -f "${env_yml_path}"
    fi
    eval "$(conda shell.bash hook)"
    conda activate one
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packes/cv2/plugin
    ;;
  win*)
    echo "Windows"
    # Create `one` env
    env_yml_path="${current_dir}/environment.yml"
    if { conda env list | grep 'one'; } >/dev/null 2>&1; then
      echo "Update 'one' environment:"
      conda env update --name one -f "${env_yml_path}"
    else
      echo "Create 'one' environment:"
      conda env create -f "${env_yml_path}"
    fi
    eval "$(conda shell.bash hook)"
    conda activate one
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packes/cv2/plugin
    ;;
  msys*)
    echo "MSYS / MinGW / Git Bash"
    ;;
  cygwin*)
    echo "Cygwin"
    ;;
  bsd*)
    echo "BSD"
     ;;
  solaris*)
    echo "Solaris"
    ;;
  *)
    echo "unknown: $OSTYPE"
    ;;
esac


# Set environment variables
# shellcheck disable=SC2162
data_dir="/data"
if [ ! -d "$data_dir" ];
then
  data_dir="${root_dir}/data"
fi
read -e -i "$data_dir" -p "Enter DATA_DIR=" input
data_dir="${input:-$data_dir}"
if [ "$data_dir" != "" ]; then
  export DATA_DIR="$data_dir"
  conda env config vars set data_dir="$data_dir"
  echo "DATA_DIR has been set to $data_dir."
else
  echo "DATA_DIR has NOT been set."
fi
if [ -d "$root_dir" ];
then
  echo "DATA_DIR=$data_dir" > "${root_dir}/pycharm.env"
fi


# Setup resilio sync
rsync_dir="${root_dir}/.sync"
mkdir -p "${rsync_dir}"
cp "IgnoreList" "${rsync_dir}/IgnoreList"
