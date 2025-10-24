#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZINF model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Zero-Shot Implicit Neural Fusion Network for Multimodal Low-Light
      Image Enhancement," arXiv 2025.
    - Code: https://github.com/phlong3105/mon

Usage:
    python predict.py --p --data "dicm, lime, mef, npe, vv, sice, fivek, darkface" --save-debug
"""

import copy

import box
import cv2
import numpy as np
import thop
import torch

import mon
import zinf
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def compute_model_stats(model: mon.nn.Module, imgsz: int = 512) -> tuple[float, float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Default: ``512``.

    Returns:
        A tuple of :math:`(flops, params)`.
    """
    image      = torch.rand(1, 1, imgsz, imgsz)
    image_lr   = model.interpolate_image(image, imgsz)
    #
    spatial    = model.create_coords(imgsz)
    patch      = model.create_patches(image_lr, model.window_size)
    #
    spatial_ff = model.ff_embedding(spatial, model.B1)
    patch_ff   = model.ff_embedding(patch,   model.B2)
    
    macs, params = thop.profile(model, inputs=(spatial_ff, patch_ff, patch_ff, patch_ff, ), verbose=False)
    flops        = 2 * macs
    # flops         = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    # params        = model.params           if hasattr(model, "params") and params == 0 else params
    # params        = parameter_count(model) if hasattr(model, "params") else params
    # params        = sum(params.values())   if isinstance(params, dict) else params
    return params, macs, flops


def benchmark(model: mon.nn.Module):
    params, macs, flops = compute_model_stats(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"MACs      : {macs:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
    
    # Model
    args.network |= {
        "iters": args.epochs,
    }
    model = zinf.ZINF(**args.network)
    model = model.to(device)
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Data I/O
    transform = A.Compose([
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
            meta  = datapoint["meta"][0]
            path  = mon.Path(meta["path"])
            image = datapoint["image"]
            image = image.to(device)
            depth = datapoint.get("depth", None)
            depth = depth.to(device) if depth is not None else None
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            outputs = model(image, depth, save_debug=args.save_debug)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs["enhanced"]
            enhanced = mon.image.to_array(enhanced)
            if args.save_debug:
                image       = mon.image.to_array(image)
                depth       = depth.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                depth       = np.clip(depth * 255, 0, 255).astype("uint8")
                depth       = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth       = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                depth       = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
                debug_image = cv2.hconcat([image, depth, enhanced])
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
            # Save Debug
            if args.save_debug:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}_debug{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(debug_image, out_path)
                for k, v in outputs.items():
                    # if k == "residual":
                    #     v = v.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    #     v = np.clip(v * 255, 0, 255).astype("uint8")
                    #     v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
                    #     v = cv2.applyColorMap(v, cv2.COLORMAP_JET)
                    #     v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                    #     out_path = out_dir / f"{path.stem}_{k}{mon.SAVE_IMAGE_EXT}"
                    #     mon.image.save_image(v, out_path)
                    if mon.image.is_image(v):
                        out_path = out_dir / f"{path.stem}_{k}{mon.SAVE_IMAGE_EXT}"
                        mon.image.save_image(v, out_path)
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
