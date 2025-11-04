#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement Ultralytics YOLOs model exporting pipeline for object detection,
classification, segmentation, orientation bounding box detection, and pose estimation.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

import box

import mon
from ultralytics import YOLO

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Export -----
def export(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Pretrained
    pretrained = None
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Export
    model = YOLO(pretrained)
    model.info()
    model.export(format="onnx")

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_predict_args(root=root_dir, model_root=root_dir)
    export(args)


if __name__ == "__main__":
    main()
