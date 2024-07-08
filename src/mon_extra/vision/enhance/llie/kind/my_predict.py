#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import socket

import click

import mon
from dataloader import *
from models import *

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = args.save_dir
    devices   = mon.set_device(args.devices)
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
        pretrain_decom = torch.load(weights / "kind_decom_net.pth")
        model.decom_net.load_state_dict(pretrain_decom)
    pretrain_restore = torch.load(weights / "kind_restore_net.pth")
    pretrain_illum   = torch.load(weights / "kind_illum_net.pth")
    model.restore_net.load_state_dict(pretrain_restore)
    model.illum_net.load_state_dict(pretrain_illum)
    model = model.to(devices)
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
    target_b = 0.70
    with torch.no_grad():
        sum_time = 0
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
                start_time  = time.time()
                bright_low  = torch.mean(image)
                bright_high = torch.ones_like(bright_low) * target_b + 0.5 * bright_low
                ratio       = torch.div(bright_high, bright_low)
                _, _, enhanced_image = model(L=image, ratio=ratio)
                # enhanced_image = enhanced_image.detach().cpu()[0]
                run_time    = (time.time() - start_time)
                output_path = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time   += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
    
# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--devices",    type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    devices   : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    weights  = weights  or args.get("weights")
    fullname = fullname or args.get("fullname")
    devices  = devices  or args.get("devices")
    imgsz    = imgsz    or args.get("imgsz")
    verbose  = verbose  or args.get("verbose")
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = weights or mon.ZOO_DIR / "vision/enhance/llie/kind"
    weights  = mon.to_list(weights)
    save_dir = save_dir or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    devices  = mon.parse_device(devices)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["data"]       = data
    args["fullname"]   = fullname
    args["save_dir"]   = save_dir
    args["devices"]    = devices
    args["imgsz"]      = imgsz
    args["resize"]     = resize
    args["benchmark"]  = benchmark
    args["save_image"] = save_image
    args["verbose"]    = verbose
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
