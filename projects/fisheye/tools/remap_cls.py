#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Remap classes from source to target dataset."""

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir

map_classes  = {
    "0":  1,  # 'bicycle',
    "1":  0,  # 'bus',
    "2":  2,  # 'car',
    "3":  3,  # 'human',
    "4":  1,  # 'motorbike',
    "5":  4,  # 'truck'
}


# ----- Remap -----
def remap_classes(data: str, split: str):
    image_dir     = data_dir / data / split / "image"
    label_old_dir = data_dir / data / split / "label_old_cls"
    label_dir     = data_dir / data / split / "label"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"[image_dir] does not exist: {image_dir}.")
    if not label_old_dir.is_dir():
        raise FileNotFoundError(f"[label_old_dir] does not exist: {label_old_dir}.")

    # Process each image
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            h, w, _ = mon.image.read_shape(image_file)
            
            # Read the old label file
            label_old_file = label_old_dir / f"{image_file.stem}.txt"
            if not label_old_file.is_txt_file(exist=True):
                continue
            bs = mon.bbox.load(path=label_old_file, fmt=mon.BBoxFormat.YOLO, imgsz=(h, w))

            # Open the new label file
            label_file = label_dir / f"{image_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_file, "w") as f:
                for b in bs:
                    c = map_classes[f"{int(b[5])}"]
                    if c == -1:  # Ignored classes
                        continue
                    f.write(f"{c} {b[0]:.32f} {b[1]:.32f} {b[2]:.32f} {b[3]:.32f}\n")


# ----- Main -----
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", required=True)
    args = parser.parse_args()
    
    remap_classes("fisheye8k", args.split)
    '''
    
    remap_classes("visdrone", "train")
