#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CoLIE model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

import copy

import box
import thop
import torch

import colie
import mon
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
def predict(args: dict | box.Box) -> str:
    window_size = args.network.window_size
    hidden_dim  = args.network.hidden_dim
    num_layers  = args.network.num_layers
    add_layer   = args.network.add_layer
    iters       = args.network.iters
    L           = args.network.L
    
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Model
    model = colie.CoLIE(
        window_size = window_size,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers,
        add_layer   = add_layer,
        iters       = iters,
        L           = L,
    )
    model = model.to(device)
    
    # Benchmark
    if args.benchmark:
        benchmark(model.model)
    
    # Data I/O
    transform = A.Compose([
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)
    
    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with (mon.create_progress_bar() as pbar):
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
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            enhanced = mon.image.to_array(enhanced)
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
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
