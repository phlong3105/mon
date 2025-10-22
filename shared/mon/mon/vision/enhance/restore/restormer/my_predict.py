#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.util import img_as_ubyte

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
from basicsr.models import create_model
from basicsr.utils.options import parse

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----

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
    # General config
    data         = args.data
    save_dir     = Path(args.save_dir)
    weights      = args.weights
    device       = args.device
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    keep_subdirs = args.keep_subdirs
    
    device    = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device    = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Override options with args
    opt           = parse(args.config, is_train=False)
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
        flops, params, avg_time = model.compute_model_stats()
        mon.log(f"FLOPs    : {flops:.4f}")
        mon.log(f"Params    : {params:.4f}")
        mon.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    mon.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Predicting
    timer = mon.Timer()
    img_multiple_of = 8
    with torch.no_grad():
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow]Predicting"
            ):
                # Input
                meta       = datapoint["meta"]
                image_path = Path(meta["path"])
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
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
                    timer.tick()
                    restored = model(input)
                    timer.tock()
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
                    timer.tick()
                    restored = E.div_(W)
                    timer.tock()
                
                restored = torch.clamp(restored, 0, 1)
                # Unpad the output
                restored = restored[:, :, :height, :width]
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0])
                
                # Save
                if save_image:
                    output_dir  = mon.rt.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                    output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if opt["image_color"] == "RGB":
                        save_img(output_path, restored)
                    else:
                        save_gray_img(output_path, restored)
        
        avg_time = float(timer.avg_time)
        mon.log(f"Average time: {avg_time}")
    



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
