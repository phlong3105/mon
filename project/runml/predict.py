#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prediction pipeline."""

from __future__ import annotations

import socket

import torch

import mon

console       = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: dict) -> str:
    # Get arguments
    fullname   = args["fullname"]
    imgsz      = mon.parse_hw(args["image_size"])
    seed       = args["seed"]
    save_dir   = args["predictor"]["default_root_dir"]
    source     = args["predictor"]["source"]
    augment    = mon.to_list(args["predictor"]["augment"])
    devices    = args["predictor"]["devices"] or "auto"
    resize     = args["predictor"]["resize"]
    benchmark  = args["predictor"]["benchmark"]
    save_image = args["predictor"]["save_image"]
    save_debug = args["predictor"]["save_debug"]
    console.rule(f"[bold red] {fullname}")
    
    # Seed
    mon.set_random_seed(seed)
    
    # Device
    devices = torch.device(("cpu" if not torch.cuda.is_available() else devices))
    
    # Benchmark
    if benchmark and torch.cuda.is_available():
        model = mon.MODELS.build(config=args["model"])
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = model,
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Model
    model: mon.Model = mon.MODELS.build(config=args["model"])
    model = model.to(devices)
    model.eval()
    
    # Data I/O
    console.log(f"[bold red] {source}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=source, dst=save_dir, denormalize=True)
    save_root = save_dir if save_dir not in [None, "None", ""] else model.root
    save_dir  = mon.Path(save_root) / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_debug:
        debug_save_dir = mon.Path(save_root) / f"{data_name}_debug"
        debug_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                images = images.to(model.device)
                h0, w0 = mon.get_image_size(images)
                input  = images.clone()
                if resize:
                    input = mon.resize(input, imgsz)
                else:
                    input = mon.resize_divisible(input, 32)
                
                # TTA (Pre)
                for aug in augment:
                    input = aug(input=input)
               
                # Forward
                if isinstance(input, torch.Tensor):
                    timer.tick()
                    output = model(input=input, augment=None, profile=False, out_index=-1)
                elif isinstance(input, list | tuple):
                    timer.tick()
                    output = []
                    for i in input:
                        o = model(input=i, augment=None, profile=False, out_index=-1)
                        o = o[-1] if isinstance(o, list | tuple) else o
                        output.append(o)
                else:
                    raise TypeError()
                timer.tock()
                
                # Forward (Debug)
                if save_debug and isinstance(input, torch.Tensor):
                    debug_output = model.forward_debug(input=input)
                else:
                    debug_output = None
                
                # TTA (Post)
                for aug in augment:
                    if aug.requires_post:
                        output = aug.postprocess(input=images, output=output)
                
                # Post-process
                output = output[-1] if isinstance(output, list | tuple) else output
                h1, w1 = mon.get_image_size(output)
                if h1 != h0 or w1 != w0:
                    output = mon.resize(output, (h0, w0))
                
                # Save
                if save_image:
                    output_path = save_dir / f"{meta['stem']}.png"
                    mon.write_image(output_path, output, denormalize=True)
                    
                    if data_writer is not None:
                        data_writer.write_batch(data=output)
                    
                    # Debug
                    if save_debug and isinstance(debug_output, dict):
                        for k, v in debug_output.items():
                            path = debug_save_dir / f"{meta['stem']}_{k}.png"
                            mon.write_image(path, v, denormalize=True)

        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
        
        return str(save_dir)
        
# endregion


# region Main

def parse_predict_args(model_root: str | mon.Path | None = None) -> dict:
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(mon.parse_predict_input_args())
    config     = input_args.get("config")
    root       = mon.Path(input_args.get("root"))
    
    # Get config args
    config = mon.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = None,
        config       = config,
    )
    args   = mon.load_config(config)
    
    # Prioritize input args --> config file args
    arch       = input_args.get("arch")
    model      = input_args.get("model")      or args.get("model_name")
    data       = input_args.get("data")       or args["predictor"]["source"]
    project    = input_args.get("project")    or args.get("project")
    variant    = input_args.get("variant")    or args.get("variant")
    fullname   = input_args.get("fullname")   or args.get("fullname")
    save_dir   = input_args.get("save_dir")   or args["predictor"]["default_root_dir"]
    weights    = input_args.get("weights")    or args["model"]["weights"]
    devices    = input_args.get("device")     or args["predictor"]["devices"]
    imgsz      = input_args.get("imgsz")      or args.get("image_size")
    resize     = input_args.get("resize")     or args.get("resize")
    benchmark  = input_args.get("benchmark")  or args.get("benchmark")
    save_image = input_args.get("save_image") or args.get("save_image")
    save_debug = input_args.get("save_debug") or args.get("save_debug")
    verbose    = input_args.get("verbose")    or args.get("verbose")
    extra_args = input_args.get("extra_args")
    
    # Parse arguments
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    weights  = None       if isinstance(weights, list | tuple) and len(weights) == 0 else weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    
    # Update arguments
    args["hostname"]    = hostname
    args["config"]      = config
    args["arch"]        = arch
    args["root"]        = root
    args["project"]     = project
    args["variant"]     = variant
    args["fullname"]    = fullname
    args["image_size"]  = imgsz
    args["verbose"]     = verbose
    args["model"]      |= {
        "root"      : save_dir,
        "fullname"  : fullname,
        "weights"   : weights,
        "loss"      : None,
        "metrics"   : None,
        "optimizers": None,
        "verbose"   : verbose,
    }
    args["predictor"]  |= {
        "default_root_dir": save_dir,
        "source"          : data,
        "devices"         : devices,
        "resize"          : resize,
        "benchmark"       : benchmark,
        "save_image"      : save_image,
        "save_debug"      : save_debug,
        "verbose"         : verbose,
    }
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config is not None and config.is_config_file():
        mon.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        
    return args


def main():
    args = parse_predict_args()
    predict(args)


if __name__ == "__main__":
    main()

# endregion
