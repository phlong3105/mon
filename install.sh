#!/bin/bash

# sudo chmod +x install.sh
# ./install.sh

clear
echo "${HOSTNAME}"

# ----- Input -----
declare -a options=("mon" "update" "cuda" "docker" "tensorrt" "ssh" "rlsync" "xanylabeling")
option="${1:-0}"

echo -e "\nAvailable options:"
for i in "${!options[@]}"; do
    printf "  [%d] %s\n" "$i" "${options[i]}"
done

read -p "Option: " -e -i "$option" option
option="${options[option]}"

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
if [ $(basename "${current_file}") == "install" ]; then
    root_dir=$(dirname "${current_dir}")
else
    root_dir="${current_dir}"
fi

# ----- Utils -----
check_gui_support() {
    if [ -n "$DISPLAY" ]; then
        # echo "GUI supported: X11 display server detected (DISPLAY=$DISPLAY)" >&2
        # echo "x11"
        return 0
    elif [ -n "$WAYLAND_DISPLAY" ]; then
        # echo "GUI supported: Wayland display server detected (WAYLAND_DISPLAY=$WAYLAND_DISPLAY)" >&2
        # echo "wayland"
        return 0
    else
        # echo "GUI not supported: No display server detected." >&2
        # echo "none"
        return 1
    fi
}

check_cuda() {
    if command -v nvcc >/dev/null 2>&1; then
        # echo "CUDA is installed. Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c 2-)"
        return 0
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # echo "NVIDIA driver is installed. CUDA Version: $(nvidia-smi | grep CUDA | awk '{print $6}')"
        return 0
    else
        # echo "CUDA is not installed or not detected."
        return 1
    fi
}

get_env_yaml_path() {
    # echo -e "\nGetting environment YAML path"
    if check_cuda; then
        echo "${root_dir}/setup/cuda.yaml"
    else
        echo "${root_dir}/setup/cpu.yaml"
    fi
}

add_bashrc_lines() {
    local lines=("$@")
    local bashrc="$HOME/.bashrc"

    echo "Checking and adding lines to $bashrc..." >&2
    for line in "${lines[@]}"; do
        if grep -Fx "$line" "$bashrc" >/dev/null; then
            echo "Line already exists: $line" >&2
        else
            echo "Adding line: $line" >&2
            echo "$line" >> "$bashrc" || {
                echo "Error: Failed to add line: $line" >&2
                echo "failed"
                return 1
            }
        fi
    done
}

add_bash_profile_lines() {
    local lines=("$@")
    local bash_profile="$HOME/.bash_profile"

    echo "Checking and adding lines to $bash_profile..." >&2
    for line in "${lines[@]}"; do
        if grep -Fx "$line" "$bash_profile" >/dev/null; then
            echo "Line already exists: $line" >&2
        else
            echo "Adding line: $line" >&2
            echo "$line" >> "$bash_profile" || {
                echo "Error: Failed to add line: $line" >&2
                echo "failed"
                return 1
            }
        fi
    done
}

# ----- OS System -----
install_ffmpeg() {
    echo -e "\nInstalling ffmpeg"
    case "$OSTYPE" in
        linux*)
            if sudo -n true 2>/dev/null; then
                sudo apt-get install -y \
                    ffmpeg '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev \
                    libgl1-mesa-glx libxrender-dev libxi-dev libxkbcommon-dev \
                    libxkbcommon-x11-dev
            else
                apt-get install -y \
                    ffmpeg '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev \
                    libgl1-mesa-glx libxrender-dev libxi-dev libxkbcommon-dev \
                    libxkbcommon-x11-dev
            fi
            ;;
        darwin*)
            brew install ffmpeg
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
}

install_imagemagick() {
    echo -e "\nInstalling imagemagick"
    case "$OSTYPE" in
        linux*)
            if sudo -n true 2>/dev/null; then
                sudo apt-get install -y imagemagick
            else
                apt-get install -y imagemagick
            fi
            ;;
        darwin*)
            brew install imagemagick
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
}

install_turbojpeg() {
    echo -e "\nInstalling ffmpeg"
    case "$OSTYPE" in
        linux*)
            if sudo -n true 2>/dev/null; then
                sudo apt-get install -y libturbojpeg
            else
                apt-get install -y libturbojpeg
            fi
            ;;
        darwin*)
            brew install jpeg-turbo
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
}

install_docker() {
    echo -e "\nInstalling Docker"
    case "$OSTYPE" in
        linux*)
            if sudo -n true 2>/dev/null; then
                sudo apt install curl
                curl https://get.docker.com | sh && sudo systemctl --now enable docker
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
                      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                sudo nvidia-ctk runtime configure --runtime=docker
                sudo systemctl restart docker
            else
                apt install curl
                curl https://get.docker.com | sh && systemctl --now enable docker
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
                      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                apt-get update
                apt-get install -y nvidia-container-toolkit
                nvidia-ctk runtime configure --runtime=docker
                systemctl restart docker
            fi
            ;;
        darwin*)
            brew install imagemagick
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
}

install_tensorrt() {
    echo -e "\nInstall TensorRT"

    if check_cuda; then
        echo "CUDA is installed. Proceeding with TensorRT installation."
    else
        echo "CUDA is not installed. Please install CUDA first."
        exit 1
    fi

    # Install TensorRT
    sudo apt-get update
    sudo apt-get install tensorrt
    sudo apt-get install onnx-graphsurgeon
    sudo apt autoremove

    # Build trtexec
    cd /usr/src/tensorrt/samples/trtexec || exit
    sudo make CUDA_INSTALL_DIR=/usr/local/cuda/bin TRT_LIB_DIR=/usr/src/tensorrt/bin
    sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/
}

install_ssh() {
    echo -e "\nInstall ssh"

    # Install ssh
    sudo apt update
    sudo apt install openssh-server
    sudo systemctl status ssh
}

setup_rlsync() {
    echo -e "\nSetting up Resilio Sync (rlsync)"
    rsync_dir="${root_dir}/.sync"
    mkdir -p "${rsync_dir}"
    cp "${root_dir}/setup/IgnoreList" "${rsync_dir}/IgnoreList"
}

setup_system() {
    install_ffmpeg
    install_imagemagick
    install_turbojpeg
    #install_ssh
    setup_rlsync
}

# ----- CUDA (System-wise) -----
install_nvidia_driver() {
    echo -e "\nInstalling NVIDIA driver"
    sudo apt update
    sudo apt upgrade
    sudo apt install gcc g++
    sudo apt install ubuntu-drivers-common
    sudo ubuntu-drivers devices
    sudo apt install nvidia-driver-570
    # sudo reboot now
}

install_cuda_toolkit() {
    echo -e "\nInstalling CUDA Toolkit"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
    # Modify .bashrc
    bashrc_lines=(
        "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}"
        "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    )
    add_bashrc_lines "${bashrc_lines[@]}"
    source ~/.bashrc
    # Manually add to ~/.bashrc
    # sudo nano ~/.bashrc
    # echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}"
    # echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    # source ~/.bashrc
    # sudo reboot now
}

# ----- Conda -----
update_conda() {
    echo -e "\nAdding 'conda' channels"
    conda config --append channels conda-forge
    conda config --append channels nvidia
    conda config --append channels pytorch

    echo -e "\nUpdating 'base' environment"
    conda update -n base -c defaults conda --y
    pip install --upgrade pip poetry
    # conda update --a --y
}

create_mon_env_linux() {
    echo -e "\nCreating 'mon' environment:"
    # Install gcc and g++
    if sudo -n true 2>/dev/null; then
        sudo apt-get install -y gcc g++
    else
        apt-get install -y gcc g++
    fi
    # Create `mon` env
    env_yaml_path=$(get_env_yaml_path)
    conda env create -f "${env_yaml_path}"
    # Modify .bashrc
    bashrc_lines=(
        # "cd '${root_dir}'"
        "conda activate mon"
    )
    add_bashrc_lines "${bashrc_lines[@]}"
    source ~/.bashrc
    # Cleanup
    rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
    echo -e "... Done"
}

create_mon_env_darwin() {
    echo -e "\nCreating 'mon' environment:"
    # Must be called before installing PyTorch Lightning
    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
    # Create `mon` env
    env_yaml_path=$(get_env_yaml_path)
    conda env create -f "${env_yaml_path}"
    # Modify .bash_profile
    bash_profile_lines=(
        # "cd '${root_dir}'"
        "conda activate mon"
    )
    add_bash_profile_lines "${bash_profile_lines[@]}"
    source ~/.bash_profile
    # Cleanup
    rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
    echo -e "... Done"
}

create_mon_env() {
    case "$OSTYPE" in
        linux*)
            create_mon_env_linux
            ;;
        darwin*)
            create_mon_env_darwin
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
}

install_mon_env() {
    echo -e "\nUpdate 'mon' library"
    eval "$(conda shell.bash hook)"
    conda activate mon
    rm -rf poetry.lock
    if check_gui_support; then
        poetry install --extras "docs gui"
    else
        poetry install --extras "docs"
    fi
    rm -rf poetry.lock
    conda update --a --y
    conda clean  --a --y
}

# ----- Tool -----
install_xanylabeling() {
    echo -e "\nInstall X-AnyLabeling"
    xanylabeling_dir="${current_dir}/tools/xanylabeling"

    # Download repo
    if [ ! -d "$xanylabeling_dir" ]; then
        git clone https://github.com/CVHub520/X-AnyLabeling.git "${xanylabeling_dir}"
    fi

    cd "${xanylabeling_dir}" || exit

    # Create conda environment
    conda create --name xanylabeling python=3.9 --y
    eval "$(conda shell.bash hook)"
    conda activate xanylabeling
    pip install -U pip
    if check_cuda; then
        pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    fi

    case "$OSTYPE" in
      linux*)
        pip install -r requirements-dev.txt
        if sudo -n true 2>/dev/null; then
            sudo apt-get install libxcb-xinerama0
        else
            apt-get install libxcb-xinerama0
        fi
        ;;
      darwin*)
          pip install -r requirements-macos-dev.txt
          ;;
      *)
          echo -e "\nunknown: $OSTYPE"
          ;;
    esac
}

# ----- Main -----
case "${option}" in
    mon)
        echo -e "\nOption: mon"
        setup_system
        update_conda
        create_mon_env
        install_mon_env
        ;;
    update)
        echo -e "\nOption: update"
        update_conda
        install_mon_env
        ;;
    cuda)
        echo -e "\nOption: cuda"
        install_nvidia_driver
        install_cuda_toolkit
        ;;
    docker)
        echo -e "\nOption: docker"
        install_docker
        ;;
    tensorrt)
        echo -e "\nOption: tensorrt"
        install_tensorrt
        ;;
    rlsync)
        echo -e "\nOption: rlsync"
        setup_rlsync
        ;;
    xanylabeling)
        echo -e "\nOption: xanylabeling"
        install_xanylabeling
        ;;
    *)
        echo -e "\nInvalid option: $option"
        exit 1
        ;;
esac

# ----- Done -----
cd "${current_dir}" || exit
exit 0
