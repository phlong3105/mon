#!/bin/bash

# Remember to install ImageMagick first:
# sudo apt-get install imagemagick
# brew install imagemagick

clear
echo "${HOSTNAME}"

# ----- Input -----
directory="/Users/longpham/Downloads/sti"

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")  # mon/tool/
root_dir=$(dirname "${current_dir}")      # mon/

# ----- Functions -----
run_on_linux() {
    cd "${directory}" || exit
    find . -type f -regex ".*\.\(bmp\|heic\|jpeg\|pgm\|png\|ppm\|webp\)" -exec mogrify -format jpg {} \; -print
    find . -type f -regex ".*\.\(bmp\|heic\|jpeg\|pgm\|png\|ppm\|webp\)" -exec rm {} \; -print
}

run_on_darwin() {
    cd "${directory}" || exit
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

# ----- Main -----
run

# ----- Done -----
exit 0
