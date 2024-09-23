#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/jinnh/GSAD

from __future__ import annotations

import argparse
import logging
import random

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

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
transform     = transforms.Lambda(lambda t: (t * 2) - 1)


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
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
        console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta        = datapoint.get("meta")
                image_path  = mon.Path(meta["path"])
                raw_img     = Image.open(image_path).convert("RGB")
                w, h        = raw_img.size[0], raw_img.size[1]
                raw_img     = transforms.Resize((h // 16 * 16, w // 16 * 16))(raw_img)
                # raw_img     = transforms.Resize(((h // 16 * 16) // 2, (w // 16 * 16) // 2))(raw_img)  # For large image
                raw_img     = transform(F.to_tensor(raw_img)).unsqueeze(0).cuda()
                
                # Infer
                timer.tick()
                diffusion.feed_data(
                    data = {
                        "LQ": raw_img,
                        "GT": raw_img,
                    }
                )
                diffusion.test(continous=False)
                timer.tock()
                
                # Post-process
                visuals     = diffusion.get_current_visuals()
                normal_img  = Metrics.tensor2img(visuals["HQ"])
                normal_img  = cv2.resize(normal_img, (w, h))
               
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    util.save_img(normal_img, str(output_path))
       
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
