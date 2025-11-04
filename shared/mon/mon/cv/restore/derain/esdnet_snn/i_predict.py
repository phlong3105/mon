#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import box
import torch
import torch.optim
import torch.nn.functional as F

import model as M
import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
from spikingjelly.activation_based import functional

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def benchmark(model: nn.Module):
    params, macs, flops = mon.nn.compute_model_stats(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"MACs      : {macs:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


def get_score_map(b, c, h, w, is_mean: bool = True) -> torch.Tensor:
    center_h = h / 2
    center_w = w / 2
    score    = torch.ones((b, c, h, w))
    if not is_mean:
        for h in range(h):
            for w in range(w):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def split_image(
    img_tensor  : torch.Tensor,
    crop_size   : int = 80,
    overlap_size: int = 8
) -> tuple[list, list]:
    b, c, h, w = img_tensor.shape
    
    h_starts = [x for x in range(0, h, crop_size - overlap_size)]
    while h_starts[-1] + crop_size >= h:
        h_starts.pop()
    h_starts.append(h - crop_size)
    
    w_starts = [x for x in range(0, w, crop_size - overlap_size)]
    while w_starts[-1] + crop_size >= w:
        w_starts.pop()
    w_starts.append(w - crop_size)
   
    starts     = []
    split_data = []
    for hs in h_starts:
        for ws in w_starts:
            c_img_data = img_tensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(c_img_data)
    return split_data, starts


def merge_image(split_data, starts, crop_size, shape=(1, 3, 80, 80)) -> torch.Tensor:
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    tot_score  = torch.zeros((b, c, h, w))
    merge_img  = torch.zeros((b, c, h, w))
    score_map  = get_score_map(b, c, crop_size, crop_size, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += score_map * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += score_map
    merge_img = merge_img / tot_score
    return merge_img


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
        state_dict = torch.load(pretrained, map_location=device, weights_only=True)
        if pretrained.is_ckpt_file(exist=True):
            state_dict = state_dict["state_dict"]
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = M.model
    functional.set_step_mode(model, step_mode="m")
    functional.set_backend(model,   backend="cupy")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Benchmark
    if args.benchmark:
        benchmark(model)
        
    # Predicting
    crop_size    = args.imgsz[0]  # 80
    overlap_size = 8              # 8
    pad_size     = 16             # 16 * 2 = 32
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image.imgsz(image)
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                image = mon.resize(image, size=args.imgsz)
            image  = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode="constant", value=0)
            image  = image.to(device)
            b, c, h1, w1 = image.shape
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            split_data, starts = split_image(image, crop_size=crop_size, overlap_size=overlap_size)
            for j, data in enumerate(split_data):
                split_data[j] = model(data).to(device)
                split_data[j] = split_data[j].cpu()
                functional.reset_net(model)
            enhanced = merge_image(split_data, starts, crop_size=crop_size, shape=(b, c, h1, w1))
            enhanced = torch.clamp(enhanced, 0, 1)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            enhanced = enhanced[:, :, pad_size:-pad_size, pad_size:-pad_size]
            if h1 != h0 or w1 != w0:
                enhanced = mon.resize(enhanced, (h0, w0))
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
