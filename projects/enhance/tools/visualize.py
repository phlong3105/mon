#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes bounding boxes on images."""

import cv2

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
run_dir      = current_file.parents[1] / "run" / "predict"


def visualize_bbox(arch: str, method: str, data: str):
    image_dir    = run_dir / arch / method / data / "pred"
    label_dir    = run_dir / arch / method / data / "pred_deim_dfine_s_coco80"
    #label_dir    = run_dir / arch / method / data / "pred_deim_dfine_s_widerface"
    classes_file = run_dir / arch / method / data / "classes.yaml"
    vis_dir 	 = run_dir / arch / method / data / "pred_deim_dfine_s_coco80_vis"
    #vis_dir      = run_dir / arch / method / data / "pred_deim_dfine_s_widerface_vis"

    if not image_dir.exists():
        raise FileNotFoundError(f"[image_dir] does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"[label_dir] does not exist: {label_dir}.")

    # Read classes from YAML file
    classes = mon.load_config(classes_file, verbose=False)
    classes = classes.get("classes", [])

    # Process each image
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow]Processing"
        ):
            # Read image
            image   = cv2.imread(str(image_file))
            image   = image[:, :, ::-1]  # Convert BGR to RGB
            h, w, _ = image.shape

            # Read YOLO label file
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue

            bs = mon.load_hbb(label_file, fmt=mon.BBoxFormat.YOLO2VOC, imgsz=(h, w))

            # Draw bounding boxes on the image
            for j, b in enumerate(bs):
                #if len(b) >= 6:
                #    l = f"{j} {int(b[4])}: {b[5]:.4f}"
                #else:
                #    l = f"{j} {int(b[4])}"
                l = f""
                c = int(b[4])
                if c >= len(classes):
                    continue
                image = mon.draw_bbox(
                    image     = image,
                    bbox      = b,
                    label     = l,
                    color     = classes[c]["color"],
                    thickness = 2,
                    fill      = False,
                )
            """
            image = cv2.putText(
                img       = image,
                text      = f"{image_file.stem}",
                org       = [50, 50],
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color     = [255, 255, 255],
                thickness = 3,
                lineType  = cv2.LINE_AA,
            )
            """
            # Save
            image	    = image[:, :, ::-1]  # Convert RGB back to BGR for saving
            output_file = vis_dir / f"{image_file.stem}.jpg"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_file), image)


if __name__ == "__main__":
    #visualize_bbox("zinf", "zinf", "exdark")
    #visualize_bbox("zinf", "zinf", "lolistreetval")

    #visualize_bbox("io"              , "image"           , "darkface")
    #visualize_bbox("zinf"            , "zinf"            , "darkface")
    #visualize_bbox("zinf"            , "zinf_view"       , "darkface")
    #visualize_bbox("colie"           , "colie"           , "darkface")
    #visualize_bbox("zerodce"         , "zerodce_siceme"  , "darkface")
    #visualize_bbox("nerco"           , "nerco_lolv1"     , "darkface")
    #visualize_bbox("lightendiffusion", "lightendiffusion", "darkface")
    #visualize_bbox("fourierdiff"     , "fourierdiff"     , "darkface")
    #visualize_bbox("zeroig"          , "zeroig"          , "darkface")

    visualize_bbox("io"              , "image"           , "lolistreetval")
    visualize_bbox("zinf"            , "zinf"            , "lolistreetval")
    visualize_bbox("colie"           , "colie"           , "lolistreetval")
    visualize_bbox("zerodce"         , "zerodce_siceme"  , "lolistreetval")
    visualize_bbox("nerco"           , "nerco_lolv1"     , "lolistreetval")
    visualize_bbox("lightendiffusion", "lightendiffusion", "lolistreetval")
    visualize_bbox("fourierdiff"     , "fourierdiff"     , "lolistreetval")
    visualize_bbox("zeroig"          , "zeroig"          , "lolistreetval")

    # visualize_bbox("io"              , "image"           , "lolistreettest")
    # visualize_bbox("zinf"            , "zinf"            , "lolistreettest")
    # visualize_bbox("colie"           , "colie"           , "lolistreettest")
    # visualize_bbox("zerodce"         , "zerodce_siceme"  , "lolistreettest")
    # visualize_bbox("nerco"           , "nerco_lolv1"     , "lolistreettest")
    # visualize_bbox("lightendiffusion", "lightendiffusion", "lolistreettest")
    # visualize_bbox("fourierdiff"     , "fourierdiff"     , "lolistreettest")
    # visualize_bbox("zeroig"          , "zeroig"          , "lolistreettest")
