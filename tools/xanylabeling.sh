#!/bin/bash

clear
echo "${HOSTNAME}"

# --- Input ---
declare -a OPTIONS=("install" "start")
OPTION="${1:-1}"

echo -e "\nAvailable OPTIONS:"
for i in "${!OPTIONS[@]}"; do
    printf "  [%d] %s\n" "$i" "${OPTIONS[i]}"
done

read -p "Option: " -e -i "$OPTION" OPTION
OPTION="${OPTIONS[OPTION]}"

# --- Directory & File ---
CURRENT_FILE=$(readlink -f "${0}")
CURRENT_DIR=$(dirname "${CURRENT_FILE}")
if [ $(basename "${CURRENT_FILE}") == "tool" ]; then
    ROOT_DIR=$(dirname "${CURRENT_DIR}")
else
    ROOT_DIR="${CURRENT_DIR}"
fi
XANYLABELING_DIR="${CURRENT_DIR}/xanylabeling"

# --- Utils ---
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

# --- Setup ---
install_xanylabeling() {
  echo -e "\nInstall X-AnyLabeling"

  # Download repo
  if [ ! -d "$XANYLABELING_DIR" ]; then
    git clone https://github.com/CVHub520/X-AnyLabeling.git
  fi
  cd "${XANYLABELING_DIR}" || exit

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

# --- Main ---
case "${OPTION}" in
    install)
        echo -e "\nOption: install"
        install_xanylabeling
        ;;
    start)
        echo -e "\nOption: start"
        eval "$(conda shell.bash hook)"
        conda activate xanylabeling
        cd "${XANYLABELING_DIR}" || exit
        python anylabeling/app.py
        ;;
    *)
        echo -e "\nInvalid OPTION: ${OPTION}"
        exit 1
        ;;
esac

# --- Done ---
exit 0
