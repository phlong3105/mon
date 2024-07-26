#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements online learning pipeline."""

from __future__ import annotations

import socket

import torch

import mon

console = mon.console


# region Online Learning

def online_learning(args: dict) -> str:
    # Get arguments
    fullname   = args["fullname"]
    imgsz      = args["image_size"]
    seed       = args["seed"]
    save_dir   = args["predictor"]["default_root_dir"]
    source     = args["predictor"]["source"]
    augment    = mon.to_list(args["predictor"]["augment"])
    devices    = args["predictor"]["devices"] or "auto"
    resize     = args["predictor"]["resize"]
    benchmark  = args["predictor"]["benchmark"]
    save_image = args["predictor"]["save_image"]
    save_debug = args["predictor"]["save_debug"]
    
    # Initialization
    console.rule(f"[bold red] {fullname}")
    
    mon.set_random_seed(seed)
    
    devices          = torch.device(("cpu" if not torch.cuda.is_available() else devices))
    model: mon.Model = mon.MODELS.build(config=args["model"])
    model            = model.to(devices)
    model.eval()
    
    # Benchmark
    if benchmark and torch.cuda.is_available():
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
        
    # Data I/O
    console.log(f"[bold red] {source}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=source, dst=save_dir, denormalize=True)
    save_dir = save_dir if save_dir not in [None, "None", ""] else model.root
    save_dir = mon.Path(save_dir)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
                input  = images.clone()
                if resize:
                    h0, w0 = mon.get_image_size(images)
                    input  = mon.resize(input, imgsz)
                
                # Forward
                timer.tick()
                output = model.fit_one(input=input)
                timer.tock()
                
                # Post-process
                output = output[-1] if isinstance(output, list | tuple) else output
                if resize:
                    output = mon.resize(output, (h0, w0))
                
                # Save
                if save_image:
                    output_path = save_dir / f"{meta['stem']}.png"
                    mon.write_image(output_path, output, denormalize=True)
                    if data_writer is not None:
                        data_writer.write_batch(data=output)
        
        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
        
        return str(save_dir)
        
# endregion


# region Main

def parse_online_args(model_root: str | mon.Path | None = None) -> dict:
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(mon.parse_train_input_args())
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
    root       = root                         or args.get("root")
    project    = input_args.get("project")    or args.get("project")
    variant    = input_args.get("variant")    or args.get("variant")
    fullname   = input_args.get("fullname")   or args.get("fullname")
    save_dir   = input_args.get("save_dir")   or args["predictor"]["default_root_dir"]
    weights    = input_args.get("weights")    or args["model"]["weights"]
    devices    = input_args.get("device")     or args["predictor"]["devices"]
    local_rank = input_args.get("local_rank")
    launcher   = input_args.get("launcher")
    epochs     = input_args.get("epochs")
    epochs     = epochs       if epochs > 0 else args["trainer"]["max_epochs"]
    steps      = input_args.get("steps")
    steps      = steps        if steps  > 0 else args["trainer"]["max_steps"]
    imgsz      = input_args.get("imgsz")      or args.get("image_size")
    resize     = input_args.get("resize")     or args.get("resize")
    benchmark  = input_args.get("benchmark")  or args.get("benchmark")
    save_image = input_args.get("save_image") or args.get("save_image")
    save_debug = input_args.get("save_debug") or args.get("save_debug")
    exist_ok   = input_args.get("exist_ok")   or args.get("exist_ok")
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
    args["root"]        = root
    args["fullname"]    = fullname
    args["image_size"]  = imgsz
    args["verbose"]     = verbose
    args["model"]      |= {
        "root"    : save_dir,
        "fullname": fullname,
        "weights" : weights,
        "verbose" : verbose,
    }
    args["trainer"]    |= {
        "default_root_dir": save_dir,
        "devices"         : devices,
        "max_epochs"      : epochs if steps is not None else None,
        "max_steps"       : steps,
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
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(save_dir))
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config is not None and config.is_config_file():
        mon.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
    
    return args


def main():
    args = parse_online_args()
    online_learning(args)


if __name__ == "__main__":
    main()

# endregion
