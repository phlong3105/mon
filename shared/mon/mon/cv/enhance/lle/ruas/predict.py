#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RUAS model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement," 2021.
    - Code: https://github.com/KarelZhang/RUAS
"""

import copy

import box
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import mon
import ruas
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, 'png')


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device          = mon.create_device(args.device)
    cudnn.benchmark = True
    cudnn.enabled   = True
    
    # Seed
    mon.set_random_seed(args.seed)
   
    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = ruas.RUAS()
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
     
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)

    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            image  = datapoint["image"]
            image  = image.to(device)
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            u_list, _ = outputs
            enhanced  = u_list[-1]
            enhanced  = mon.image.to_array(enhanced)
            debug     = u_list[-2]
            debug     = mon.image.to_array(debug)
            h1, w1    = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
                debug    = cv2.resize(debug, (w0, h0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
                """
                # out_path.parent.mkdir(parents=True, exist_ok=True)
                # save_images(u_list[-1], str(out_path))
                # save_images(u_list[-1], str(args.output_dir / "lol" / u_name))
                # save_images(u_list[-2], str(args.output_dir / "dark" / u_name))
                if args.model == "lol":
                    save_images(u_list[-1], u_path)
                elif args.model == "upe" or args.model == "dark":
                    save_images(u_list[-2], u_path)
                """

            if args.save_debug:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}_dark{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(debug, out_path)
                # out_path.parent.mkdir(parents=True, exist_ok=True)
                # save_images(u_list[-2], str(out_path))
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.rt.parse_cli_args(root=root_dir)
    data = mon.utils.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.rt.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
