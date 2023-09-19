#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/pvnieo/Low-light-Image-Enhancement

from __future__ import annotations

import argparse
import time
from argparse import RawTextHelpFormatter

import cv2

import mon
from exposure_enhancement import enhance_image_exposure

console = mon.console


def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--data",       type=str,   default="./demo/", help="folder path to test images.")
    parser.add_argument("--weights",    type=str,   default="./checkpoint.pth")
    parser.add_argument("--image-size", type=int,   default=512)
    parser.add_argument("--gamma",      type=float, default=0.6,       help="Gamma correction parameter.")
    parser.add_argument("--lambda_",    type=float, default=0.15,      help="The weight for balancing the two terms in the illumination refinement optimization objective.")
    parser.add_argument("--lime",       action="store_true",           help="Use the LIME method. By default, the DUAL method is used.")
    parser.add_argument("--resize",     action="store_true")
    parser.add_argument("--sigma",      type=int,   default=3,         help="Spatial standard deviation for spatial affinity based Gaussian weights.")
    parser.add_argument("--bc",         type=float, default=1,         help="Parameter for controlling the influence of Mertens's contrast measure.")
    parser.add_argument("--bs",         type=float, default=1,         help="Parameter for controlling the influence of Mertens's saturation measure.")
    parser.add_argument("--be",         type=float, default=1,         help="Parameter for controlling the influence of Mertens's well exposedness measure.")
    parser.add_argument("--eps",        type=float, default=1e-3,      help="Constant to avoid computation instability.")
    parser.add_argument("--output-dir", type=str,   default=mon.RUN_DIR/"predict/lime")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    #
    image_paths = list(args.data.rglob("*"))
    image_paths = [path for path in image_paths if path.is_image_file()]
    sum_time    = 0
    with mon.get_progress_bar() as pbar:
        for _, image_path in pbar.track(
            sequence    = enumerate(image_paths),
            total       = len(image_paths),
            description = f"[bright_yellow] Inferring"
        ):
            # console.log(image_path)
            image          = cv2.imread(str(image_path))
            h, w, c        = image.shape
            if args.resize:
                image = cv2.resize(image, (args.image_size, args.image_size))
            start_time     = time.time()
            enhanced_image = enhance_image_exposure(
                im      = image,
                gamma   = args.gamma,
                lambda_ = args.lambda_,
                dual    = not args.lime,
                sigma   = args.sigma,
                bc      = args.bc,
                bs      = args.bs,
                be      = args.be,
                eps     = args.eps
            )
            run_time       = (time.time() - start_time)
            if args.resize:
                enhanced_image = cv2.resize(enhanced_image, (w, h))
            result_path    = args.output_dir / image_path.name
            cv2.imwrite(str(result_path), enhanced_image)
            sum_time      += run_time
    avg_time = float(sum_time / len(image_paths))
    console.log(f"Average time: {avg_time}")
    

if __name__ == "__main__":
    main()
