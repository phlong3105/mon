#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/jinnh/GSAD

from __future__ import annotations

import argparse
import logging
import random
import socket
import time

import click
import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

import core.logger as Logger
import core.metrics as Metrics
import model as Model
import mon
import options.options as option
from utils import util

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
transform     = transforms.Lambda(lambda t: (t * 2) - 1)


# region Predict

def predict(args: argparse.Namespace):
    # General config
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = args.save_dir
    device    = mon.set_device(args.device)
    launcher  = args.launcher
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Override options with args
    opt           = Logger.parse(args)
    opt           = Logger.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    opt["phase"]  = "test"
    opt["device"] = device
    
    # Distributed training settings
    opt["dist"] = False
    rank = -1
    # print("Disabled distributed training.")
    
    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        util.setup_logger("val", opt["path"]["log"], "val_" + opt["name"], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger("base")
        # logger.info(option.dict2str(opt))
    util.setup_logger("base", opt["path"]["log"], "train", level=logging.INFO, screen=True)
    logger = logging.getLogger("base")
    
    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    
    # Random seed
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    # if rank <= 0:
        # logger.info("Random seed: {}".format(seed))
    util.set_random_seed(seed)
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    # Model
    opt["path"]["resume_state"] = str(weights)
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["val"], schedule_phase="val")
    # logger.info("Initial Model Finished")
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = diffusion.measure_efficiency_score(image_size=imgsz)
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path  = meta["path"]
                raw_img     = Image.open(image_path).convert("RGB")
                w, h        = raw_img.size[0], raw_img.size[1]
                raw_img     = transforms.Resize((h // 16 * 16, w // 16 * 16))(raw_img)
                # raw_img     = transforms.Resize(((h // 16 * 16) // 2, (w // 16 * 16) // 2))(raw_img)  # For large image
                raw_img     = transform(F.to_tensor(raw_img)).unsqueeze(0).cuda()
                start_time  = time.time()
                diffusion.feed_data(
                    data = {
                        "LQ": raw_img,
                        "GT": raw_img,
                    }
                )
                diffusion.test(continous=False)
                run_time    = (time.time() - start_time)
                visuals     = diffusion.get_current_visuals()
                normal_img  = Metrics.tensor2img(visuals["HQ"])
                normal_img  = cv2.resize(normal_img, (w, h))
                output_path = save_dir / image_path.name
                util.save_img(normal_img, str(output_path))
                sum_time   += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",           type=str, default=None, help="Project root.")
@click.option("--config",         type=str, default=None, help="Model config.")
@click.option("--weights",        type=str, default=None, help="Weights paths.")
@click.option("--model",          type=str, default=None, help="Model name.")
@click.option("--data",           type=str, default=None, help="Source data directory.")
@click.option("--fullname",       type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",       type=str, default=None, help="Optional saving directory.")
@click.option("--device",         type=str, default=None, help="Running device.")
@click.option("--local-rank",     type=int, default=0)
@click.option("--launcher",       type=click.Choice(["none", "pytorch"]), default="none", help="Job launcher.")
@click.option("--phase",          type=click.Choice(["train", "val"]), default="train", help="Run either train(training) or val(generation).")
@click.option("--imgsz",          type=int, default=None, help="Image sizes.")
@click.option("--resize",         is_flag=True)
@click.option("--benchmark",      is_flag=True)
@click.option("--save-image",     is_flag=True)
@click.option("--tfboard",        is_flag=True)
@click.option("--debug",          is_flag=True)
@click.option("--enable_wandb",   is_flag=True)
@click.option("--log-wandb-ckpt", is_flag=True)
@click.option("--log-eval",       is_flag=True)
@click.option("--verbose",        is_flag=True)
def main(
    root          : str,
    config        : str,
    weights       : str,
    model         : str,
    data          : str,
    fullname      : str,
    save_dir      : str,
    device        : str,
    local_rank    : int,
    launcher      : str,
    phase         : str,
    imgsz         : int,
    resize        : bool,
    benchmark     : bool,
    save_image    : bool,
    tfboard       : bool,
    debug         : bool,
    enable_wandb  : bool,
    log_wandb_ckpt: bool,
    log_eval      : bool,
    verbose       : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    weights  = weights  or args.get("weights")
    fullname = fullname or args.get("fullname")
    device   = device   or args.get("device")
    imgsz    = imgsz    or args.get("imgsz")
    verbose  = verbose  or args.get("verbose")
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    save_dir = save_dir or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args["root"]           = root
    args["config"]         = config
    args["weights"]        = weights
    args["model"]          = model
    args["data"]           = data
    args["fullname"]       = fullname
    args["save_dir"]       = save_dir
    args["device"]         = device
    args["local_rank"]     = local_rank
    args["launcher"]       = launcher
    args["phase"]          = phase
    args["imgsz"]          = imgsz
    args["resize"]         = resize
    args["benchmark"]      = benchmark
    args["save_image"]     = save_image
    args["tfboard"]        = tfboard
    args["debug"]          = debug
    args["enable_wandb"]   = enable_wandb
    args["log_wandb_ckpt"] = log_wandb_ckpt
    args["log_eval"]       = log_eval
    args["verbose"]        = verbose
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
