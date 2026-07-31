#!/bin/bash

# Remember to install ffmpeg first:
# sudo apt-get install ffmpeg
# brew install ffmpeg

clear
echo "${HOSTNAME}"

# --- Input ---
DIRECTORY="/home/longpham/Downloads"

# --- Directory & File ---
CURRENT_FILE=$(readlink -f "${0}")
CURRENT_DIR=$(dirname "${CURRENT_FILE}")  # mon/tool/
ROOT_DIR=$(dirname "${CURRENT_DIR}")      # mon/

# --- Functions ---
run_on_linux() {
    cd "${DIRECTORY}" || exit
    for i in $(find . -type f -regex ".*\.\(mp4\|MP4\|avi\|m4v\|mkv\|mov\|mpeg\|mpg\|wmv\)" | sort -h); do
        echo "Processing file: ${i}"
        ffmpeg \
            -i "$i" \
            -c:v libx265 \
            -c:a copy \
            -tag:v hvc1 \
            "${i%.*}_convert.mp4";
        rm "$i"
        mv "${i%.*}_convert.mp4" "${i%.*}.mp4"
    done
}

run_on_darwin() {
    cd "${DIRECTORY}" || exit
    for i in $(find . -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.webm" -o -iname "*.mpeg" -o -iname "*.mpg" -o -iname "*.3gp" -o -iname "*.m4v" \) | sort -h); do
        echo "Processing file: ${i}"
        ffmpeg \
            -i "$i" \
            -c:v libx265 \
            -c:a copy \
            -tag:v hvc1 \
            "${i%.*}_convert.mp4";
        rm "$i"
        mv "${i%.*}_convert.mp4" "${i%.*}.mp4"
    done
}

run() {
    case "$OSTYPE" in
    linux*)
        run_on_linux
        ;;
    darwin*)
        run_on_darwin
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

# --- Main ---
run

# --- Done ---
exit 0
