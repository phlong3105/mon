#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket
import time

import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.util import img_as_ubyte

import mon
from basicsr.models import create_model
from basicsr.utils.options import parse

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def load_img(filepath: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)


def load_gray_img(filepath: str) -> np.ndarray:
    return np.expand_dims(cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE), axis=2)


def save_img(filepath: str, image: np.ndarray):
    cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray_img(filepath: str, image: np.ndarray):
    cv2.imwrite(str(filepath), image)


def get_weights_and_parameters(task, parameters):
    if task == "Motion_Deblurring":
        weights = os.path.join("motion_deblurring", "pretrained_models", "motion_deblurring.pth")
    elif task == "Single_Image_Defocus_Deblurring":
        weights = os.path.join("defocus_deblurring", "pretrained_models", "single_image_defocus_deblurring.pth")
    elif task == "Deraining":
        weights = os.path.join("deraining", "pretrained_models", "deraining.pth")
    elif task == "Real_Denoising":
        weights = os.path.join("denoising", "pretrained_models", "real_denoising.pth")
        parameters["LayerNorm_type"] = "BiasFree"
    elif task == "Gaussian_Color_Denoising":
        weights = os.path.join("denoising", "pretrained_models", "gaussian_color_denoising_blind.pth")
        parameters["LayerNorm_type"] = "BiasFree"
    elif task == "Gaussian_Gray_Denoising":
        weights = os.path.join("denoising", "pretrained_models", "gaussian_gray_denoising_blind.pth")
        parameters["inp_channels"]   = 1
        parameters["out_channels"]   = 1
        parameters["LayerNorm_type"] = "BiasFree"
    return weights, parameters


def predict(args: argparse.Namespace):
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = mon.Path(args.save_dir)
    device    = args.device
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Override options with args
    opt           = parse(args.opt, is_train=False)
    opt["device"] = device
    
    # Load model
    '''
    parameters = {
        "inp_channels"         : 3,
        "out_channels"         : 3,
        "dim"                  : 48,
        "num_blocks"           : [4, 6, 6, 8],
        "num_refinement_blocks": 4,
        "heads"                : [1, 2, 4, 8],
        "ffn_expansion_factor" : 2.66,
        "bias"                 : False,
        "LayerNorm_type"       : "WithBias",
        "dual_pixel_task"      : False,
    }
    weights, parameters = get_weights_and_parameters(task, parameters)
    load_arch  = run_path(os.path.join("basicsr", "models", "archs", "restormer_arch.py"))
    model      = load_arch["Restormer"](**parameters)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["params"])
    '''
    opt["path"]["pretrain_network_g"] = str(weights)
    model = create_model(opt)
    model.to(device)
    model.eval()
    
    # Measure efficiency score
    if benchmark:
        flops, params, avg_time = model.measure_efficiency_score(image_size=imgsz)
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    img_multiple_of = 8
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                 
                image_path = meta["path"]
                if opt["image_color"] == "RGB":
                    image = load_gray_img(image_path)
                else:
                    image = load_img(image_path)
                input = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Pad the input if not_multiple_of 8
                height, width = input.shape[2], input.shape[3]
                H     = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
                W     = ((width  + img_multiple_of) // img_multiple_of) * img_multiple_of
                pad_h = H - height if height % img_multiple_of != 0 else 0
                pad_w = W - width  if width  % img_multiple_of != 0 else 0
                input = F.pad(input, (0, pad_w, 0, pad_h), "reflect")
                
                if args.tile is None:
                    # Test on the original resolution image
                    start_time = time.time()
                    restored = model(input)
                    run_time = (time.time() - start_time)
                else:
                    # Test the image tile by tile
                    b, c, h, w   = input.shape
                    tile         = min(args.tile, h, w)
                    assert tile % 8 == 0, "tile size should be multiple of 8"
                    tile_overlap = args.tile_overlap
                    stride       = tile - tile_overlap
                    h_idx_list   = list(range(0, h - tile, stride)) + [h - tile]
                    w_idx_list   = list(range(0, w - tile, stride)) + [w - tile]
                    E            = torch.zeros(b, c, h, w).type_as(input)
                    W            = torch.zeros_like(E)
                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch       = input[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                            out_patch      = model(in_patch)
                            out_patch_mask = torch.ones_like(out_patch)
                            E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                            W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                    start_time = time.time()
                    restored   = E.div_(W)
                    run_time   = (time.time() - start_time)
                
                restored = torch.clamp(restored, 0, 1)
                # Unpad the output
                restored = restored[:, :, :height, :width]
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0])
                
                output_path = save_dir / image_path.name
                if opt["image_color"] == "RGB":
                    save_img(output_path, restored)
                else:
                    save_gray_img(output_path, restored)
                sum_time += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
    
# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",         type=str, default=None, help="Project root.")
@click.option("--config",       type=str, default=None, help="Model config.")
@click.option("--weights",      type=str, default=None, help="Weights paths.")
@click.option("--model",        type=str, default=None, help="Model name.")
@click.option("--data",         type=str, default=None, help="Source data directory.")
@click.option("--fullname",     type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",     type=str, default=None, help="Optional saving directory.")
@click.option("--device",       type=str, default=None, help="Running devices.")
@click.option("--imgsz",        type=int, default=None, help="Image sizes.")
@click.option("--tile",         type=int, default=None, help="Tile size (e.g 720). None means testing on the original resolution image.")
@click.option("--tile-overlap", type=int, default=32,   help="Overlapping of different tiles.")
@click.option("--resize",       is_flag=True)
@click.option("--benchmark",    is_flag=True)
@click.option("--save-image",   is_flag=True)
@click.option("--verbose",      is_flag=True)
def main(
    root        : str,
    config      : str,
    weights     : str,
    model       : str,
    data        : str,
    fullname    : str,
    save_dir    : str,
    device      : str,
    imgsz       : int,
    tile        : int,
    tile_overlap: int,
    resize      : bool,
    benchmark   : bool,
    save_image  : bool,
    verbose     : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root or args.get("root")
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    save_dir = save_dir or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args = {
        "root"        : root,
        "config"      : config,
        "opt"         : config,
        "model"       : model,
        "data"        : data,
        "fullname"    : fullname,
        "save_dir"    : save_dir,
        "weights"     : weights,
        "device"      : device,
        "imgsz"       : imgsz,
        "tile"        : tile,
        "tile_overlap": tile_overlap,
        "resize"      : resize,
        "benchmark"   : benchmark,
        "save_image"  : save_image,
        "verbose"     : verbose,
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
