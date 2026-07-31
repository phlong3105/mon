#!/bin/bash

# Remember to install ImageMagick first:
# sudo apt-get install imagemagick
# brew install imagemagick

clear
echo "${HOSTNAME}"

# --- Input ---
DIRECTORY="/home/longpham/00_inbox/mon/projects/aic26_cross_city/data/aic26_cross_city/x/bmd45/val"

# --- Directory & File ---
CURRENT_FILE=$(readlink -f "${0}")
CURRENT_DIR=$(dirname "${CURRENT_FILE}")  # mon/tool/
ROOT_DIR=$(dirname "${CURRENT_DIR}")      # mon/

# --- Functions ---
run_on_linux() {
    cd "${DIRECTORY}" || exit
    find . -type f -regex ".*\.\(bmp\|heic\|jpeg\|pgm\|png\|ppm\|webp\)" -exec mogrify -format jpg {} \; -print
    find . -type f -regex ".*\.\(bmp\|heic\|jpeg\|pgm\|png\|ppm\|webp\)" -exec rm {} \; -print
}

run_on_darwin() {
    cd "${DIRECTORY}" || exit
    find . -type f \( -iname "*.bmp" -o -iname "*.heic" -o -iname "*.jpeg" -o -iname "*.pgm" -o -iname "*.png" -o -iname "*.ppm" -o -iname "*.webp" \) -exec mogrify -format jpg {} \; -print
    find . -type f \( -iname "*.bmp" -o -iname "*.heic" -o -iname "*.jpeg" -o -iname "*.pgm" -o -iname "*.png" -o -iname "*.ppm" -o -iname "*.webp" \) -exec rm {} \; -print
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
