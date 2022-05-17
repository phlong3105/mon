#!/bin/bash

# Install:
# chmod +x install.sh
# conda init bash
# ./install.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$(dirname "$current_dir")
one_dir=$(dirname "$root_dir")
if [ "$(basename -- $one_dir)" == "projects" ]; then
  one_dir=$(dirname "$one_dir")
fi


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
    # Create `base` env
    env_yml_path="${current_dir}/environment_docker.yml"
    conda install python=3.9
    if { conda env list | grep 'base'; } >/dev/null 2>&1; then
      echo "Update 'base' environment:"
      conda env update --name base -f "${env_yml_path}"
    else
      echo "Create 'base' environment:"
      conda env create -f "${env_yml_path}"
    fi
    eval "$(conda shell.bash hook)"
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packes/cv2/plugin
    ;;
  darwin*)
    echo "MacOS"
    # Create `base` env
    env_yml_path="${current_dir}/environment_macos.yml"
    if { conda env list | grep 'base'; } >/dev/null 2>&1; then
      echo "Update 'base' environment:"
      conda env update --name base -f "${env_yml_path}"
    else
      echo "Create 'base' environment:"
      conda env create -f "${env_yml_path}"
    fi
    eval "$(conda shell.bash hook)"
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packes/cv2/plugin
    ;;
  win*)
    echo "Windows"
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
datasets_dir="/datasets"
if [ ! -d "$datasets_dir" ];
then
  datasets_dir="${one_dir}/datasets"
fi
read -e -i "$datasets_dir" -p "Enter DATASETS_DIR=" input
datasets_dir="${input:-$datasets_dir}"
if [ "$datasets_dir" != "" ]; then
  export DATASETS_DIR="$datasets_dir"
  conda env config vars set datasets_dir="$datasets_dir"
  echo "DATASETS_DIR has been set to $datasets_dir."
else
  echo "DATASETS_DIR has NOT been set."
fi
if [ -d "$root_dir" ];
then
  echo "DATASETS_DIR=$datasets_dir" > "${root_dir}/pycharm.env"
fi
