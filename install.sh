#!/bin/bash

# Usage:
# chmod +x install.sh
# 
# conda init bash
# bash -l install.sh
#
# conda init zsh
# zsh -i install.sh


# region Functions

check_display_server() {
    # Function to check for the display server
    if [ -z "$DISPLAY" ]; then
        echo "No display server found. GUI applications may not run."
        return 1
    else
        echo "Display server is running on $DISPLAY."
        return 0
    fi
}

add_channels() {
    echo -e "\nAdding channels"
    conda config --append channels conda-forge
    conda config --append channels nvidia
    conda config --append channels pytorch
    echo -e "... Done"
}

update_base_env() {
    echo -e "\nUpdating 'base' environment:"
    conda init bash
    conda update -n base -c defaults conda --y
    conda update --a --y
    pip install --upgrade pip
    pip install poetry
    echo -e "... Done"
}

read_user_input() {
    install_type=$1
    read -e -i "$install_type" -p "Machine Type [gui, console]: " install_type
    echo "$install_type"
}

install_on_linux() {
    echo -e "\nLinux / WSL"
    # Create `mon` env
    if [ "$install_type" == "gui" ]; then
        env_yml_path="${current_dir}/env/linux_gui.yaml"
    elif [ "$install_type" == "console" ]; then
        env_yml_path="${current_dir}/env/linux_console.yaml"
    else
        echo -e "\nInvalid machine type: $install_type"
    fi

    if { conda env list | grep 'mon'; } >/dev/null 2>&1; then
        echo -e "\nUpdating 'mon' environment:"
        conda env update --name mon -f "${env_yml_path}"
        echo -e "... Done"
    else
        echo -e "\nCreating 'mon' environment:"
        if sudo -n true 2>/dev/null; then
            sudo apt-get install gcc g++
        else
            apt-get install gcc g++
        fi
        conda env create -f "${env_yml_path}"
        echo "conda activate mon" >> ~/.bashrc
        source ~/.bashrc
        echo -e "... Done"
    fi

    # Install FFMPEG
    if sudo -n true 2>/dev/null; then
        sudo apt-get install ffmpeg
        sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
    else
        apt-get install ffmpeg
        apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
    fi
}

install_on_darwin() {
    echo -e "\nMacOS"
    # Must be called before installing PyTorch Lightning
    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
    # Create `mon` env
    env_yml_path="${current_dir}/env/macos.yaml"
    if { conda env list | grep 'mon'; } >/dev/null 2>&1; then
        echo -e "\nUpdating 'mon' environment:"
        conda env update --name mon -f "${env_yml_path}"
        echo -e "... Done"
    else
        echo -e "\nCreating 'mon' environment:"
        conda env create -f "${env_yml_path}"
        echo "conda activate mon" >> ~/.bash_profile
        source ~/.bash_profile
        echo -e "... Done"
    fi
    # Install FFMPEG
    brew install ffmpeg
}

install_third_party_library() {
    echo -e "\nInstall third-party library"
    eval "$(conda shell.bash hook)"
    conda activate mon
    rm -rf poetry.lock
    rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
    poetry install --extras "dev"
    rm -rf poetry.lock
    conda update --a --y
    conda clean --a --y
}

setup_data_dir() {
    echo -e "\nSetting DATA_DIR"
    root_dir=$1
    data_dir="${root_dir}/data"
    read -e -i "$data_dir" -p "Enter DATA_DIR=" input
    data_dir="${input:-$data_dir}"
    if [ "$data_dir" != "" ]; then
        export DATA_DIR="$data_dir"
        conda env config vars set data_dir="$data_dir"
        echo -e "\nDATA_DIR has been set to $data_dir."
    else
        echo -e "\nDATA_DIR has NOT been set."
    fi
    # if [ -d "$root_dir" ]; then
    #     echo -e "\nDATA_DIR=$data_dir" > "${root_dir}/pycharm.env"
    # fi
    echo -e "... Done"
}

setup_resilio_sync() {
    rsync_dir="${root_dir}/.sync"
    mkdir -p "${rsync_dir}"
    cp "env/IgnoreList" "${rsync_dir}/IgnoreList"
    echo -e "... Done"
}

# endregion


# region Main

echo "$HOSTNAME"

# Directory
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$current_dir

install_type=$(read_user_input "${1:-"gui"}")

# Update 'base' env
add_channels
update_base_env

# Install 'mon' env
echo -e "\nInstalling 'mon' environment:"
case "$OSTYPE" in
    linux*)
        install_on_linux
        ;;
    darwin*)
        install_on_darwin
        ;;
    win*)
        echo -e "\nWindows"
        ;;
    msys*)
        echo -e "\nMSYS / MinGW / Git Bash"
        ;;
    cygwin*)
        echo -e "\nCygwin"
        ;;
    bsd*)
        echo -e "\nBSD"
        ;;
    solaris*)
        echo -e "\nSolaris"
        ;;
    *)
        echo -e "\nunknown: $OSTYPE"
        ;;
esac

install_third_party_library

# Setup environment variables
# shellcheck disable=SC2162
export PYTHONDONTWRITEBYTECODE=1
setup_data_dir "$root_dir"
setup_resilio_sync "$root_dir"

# endregion
