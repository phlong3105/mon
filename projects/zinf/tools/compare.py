#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import cv2
import numpy as np

import mon

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[1]
data_dir     = root_dir / "data"
run_dir      = root_dir / "run"

models = [
    # "zinf_siren",
    # "zinf_siren_lbfgs",
    # "zinf_isiren",
    # "zinf_isiren_lbfgs",
    # "zinf_dam_siren",
    # "zinf_dam_siren_lbfgs",
    "zinf_dam_isiren",
    "zinf_dam_isiren_lbfgs",
]


def compare(data: str) -> str:
    if data == "sice":
        input_dir = data_dir / data / "sice_lr" / "test" / "image_under"
    else:
        input_dir = data_dir / data / "test" / "image"
    colie_dir   =  run_dir / "predict" / "colie" / "colie" / data / "pred"
    model_dirs  = [run_dir / "predict" / "zinf"  / m       / data / "pred" for m in models]
    model_names = ["colie"] + models
    
    input_files = sorted(list(input_dir.glob("*")))
    for input_file in input_files:
        colie_file   = colie_dir / input_file.name
        model_files  = [colie_file] + [md / input_file.name for md in model_dirs]
        
        input_image  = cv2.imread(str(input_file))
        empty_image  = np.zeros((input_image.shape[0], input_image.shape[1], 3), np.uint8)
        concat_image = cv2.vconcat([input_image, empty_image])
        cv2.putText(concat_image, "Input", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        for i, model_file in enumerate(model_files):
            model_image  = cv2.imread(str(model_file))
            diff_image   = cv2.absdiff(model_image, input_image)
            gray_diff    = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
            norm_diff    = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)
            heatmap      = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
            cv2.putText(model_image, model_names[i], (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            concat_image = cv2.hconcat([concat_image, cv2.vconcat([model_image, heatmap])])
        
        concat_image = cv2.resize(concat_image, None, fx=0.5, fy=0.5)
        cv2.imshow("Compare", concat_image)
        cv2.waitKey(0)


# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="zinf_siren_d")
    parser.add_argument("--data",  type=str, default="dicm")
    args = parser.parse_args()
    # compare(args.model, args.data)
    compare(args.data)


if __name__ == "__main__":
    main()
