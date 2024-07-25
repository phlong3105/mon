#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import mon
from dataloader import *
from models import *

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # args.input_dir  = mon.Path(args.input_dir)
    # args.output_dir = mon.Path(args.output_dir)
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # console.log(f"Data: {args.input_dir}")
    
    # Model
    args["noDecom"] = True
    model = KinD()
    if args["noDecom"] is False:
        pretrain_decom = torch.load(weights / "kind_lol_v1_decom.pth")
        model.decom_net.load_state_dict(pretrain_decom)
    pretrain_restore = torch.load(weights / "kind_lol_v1_restore.pth")
    pretrain_illum   = torch.load(weights / "kind_lol_v1_illum.pth")
    model.restore_net.load_state_dict(pretrain_restore)
    model.illum_net.load_state_dict(pretrain_illum)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer    = mon.Timer()
    target_b = 0.70
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path  = meta["path"]
                image       = Image.open(image_path)
                image       = np.asarray(image, np.float32).transpose((2, 0, 1)) / 255.0
                image       = torch.from_numpy(image).float()
                image       = image.cuda().unsqueeze(0)
                timer.tick()
                bright_low  = torch.mean(image)
                bright_high = torch.ones_like(bright_low) * target_b + 0.5 * bright_low
                ratio       = torch.div(bright_high, bright_low)
                _, _, enhanced_image = model(L=image, ratio=ratio)
                # enhanced_image = enhanced_image.detach().cpu()[0]
                timer.tock()
                output_path = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
    
# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()
# endregion
