#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements InDi-Deband model prediction pipeline for image/video debanding

References:
    - Paper:
    - Code: https://github.com/ksasso1028/indi-debanding
"""

import copy

import box
import cv2
import thop
import torch

import indi_deband
import mon
import mon.training.albumentations as A

mon.preload()

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
    patches      = torch.rand(imgsz, imgsz, 49).to(mon.get_model_device(model))
    coords       = torch.rand(imgsz, imgsz,  2).to(mon.get_model_device(model))
    macs, params = thop.profile(model, inputs=(patches, coords,), verbose=False)
    flops        = 2 * macs
    return params, macs, flops


def benchmark(model: mon.nn.Module):
    params, macs, flops = compute_model_stats(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"MACs      : {macs:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)
    
    # Device
    device = mon.create_device(args.device)
    
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
    model = indi_deband.InDiDeband(weights=pretrained, **args.network)
    model = model.to(device)
    model.eval()
    steps = args.indi.steps
    
    # Benchmark
    if args.benchmark:
        mon.metrics.benchmark(model)
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.build_dataloader(args.data, args.root, transform)
    
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
            outputs = indi_deband.sample(model, image, steps)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            enhanced = mon.image.to_array(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save(enhanced, out_path)
    timers.total.tock()
    
    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.parse_cli_args(root=root_dir)
    data = mon.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
