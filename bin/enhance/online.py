#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prediction pipeline."""

from __future__ import annotations

import importlib
import random
import socket
import time
from typing import Any

import click
import torch
import torchvision

import mon

console = mon.console


# region Function

def predict(args: dict):
    # Get arguments
    model_name    = args["model"]["name"]
    variant       = args["model"]["variant"]
    weights       = args["model"]["weights"]
    optimizer     = args["model"]["optimizer"]
    input_dir     = args["datamodule"]["root"]
    image_size    = args["datamodule"]["image_size"]
    resize        = args["datamodule"]["resize"]
    devices       = args["predictor"]["devices"] or "auto"
    benchmark     = args["predictor"]["benchmark"]
    output_dir    = args["predictor"]["output_dir"]
    save_image    = args["predictor"]["save_image"]
    verbose       = args["predictor"]["verbose"]

    # Initialization
    variant       = variant if variant not in [None, "", "none"] else None
    model_variant = f"{model_name}-{variant}" if variant is not None else f"{model_name}"
    console.rule(f"[bold red] {model_variant}")

    devices = "cpu" if not torch.cuda.is_available() else devices
    devices = torch.device(devices)

    model: mon.Model = mon.MODELS.build(config=args["model"])
    state_dict  = torch.load(weights, map_location=devices)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    model.phase = mon.ModelPhase.TRAINING
    optimizer   = mon.OPTIMIZERS.build(net=model, config=optimizer[0])

    # Measure efficiency score
    if benchmark and torch.cuda.is_available():
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = model,
            image_size = image_size,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
     
    # Data
    input_dir = mon.Path(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"{input_dir}")
    if input_dir.is_video_file():
        image_loader = mon.VideoLoaderCV(source=input_dir, to_rgb=True, to_tensor=True, normalize=True)
        video_writer = mon.VideoWriterCV(
            destination = output_dir / input_dir.stem,
            image_size  = [480, 640],
            frame_rate  = 30,
            fourcc      = "mp4v",
            save_image  = False,
            denormalize = True,
            verbose     = False,
        )
    else:
        image_loader = mon.ImageLoader(source=input_dir, to_rgb=True, to_tensor=True, normalize=True)
        video_writer = None

    #
    h, w = mon.get_hw(image_size)
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, indexes, files, rel_paths in pbar.track(
                sequence    = enumerate(image_loader),
                total       = len(image_loader),
                description = f"[bright_yellow] Inferring"
            ):
                if resize:
                    h0, w0   = mon.get_image_size(images)
                    images   = mon.resize(input=images, size=[h, w])
                input        = images.to(model.device)
                start_time   = time.time()
                output, loss = model.forward_loss(input=input, target=None)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
                optimizer.step()
                
                run_time    = time.time() - start_time
                output      = output[-1] if isinstance(output, (list, tuple)) else output
                if resize:
                    output  = mon.resize(input=images, size=[h0, w0])

                if save_image:
                    output_path = output_dir / f"{files[0].stem}.png"
                    torchvision.utils.save_image(output, str(output_path))
                if input_dir.is_video_file():
                    video_writer.write_batch(images=output)
                sum_time += run_time
        avg_time = float(sum_time / len(image_loader))
        console.log(f"Average time: {avg_time}")


@click.command(context_settings=dict(
    ignore_unknown_options = True,
    allow_extra_args       = True,
))
@click.option("--config",     type=click.Path(exists=False), default=None,                  help="The training config to use.")
@click.option("--input-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR,          help="Source data directory.")
@click.option("--output-dir", type=click.Path(exists=False), default=mon.RUN_DIR/"predict", help="Save results location.")
@click.option("--name",       type=str,                      default=None,                  help="Model name.")
@click.option("--variant",    type=str,                      default=None,                  help="Model variant.")
@click.option("--data",       type=str,                      default=None,                  help="Training dataset name.")
@click.option("--root",       type=click.Path(exists=False), default=mon.RUN_DIR/"predict", help="Save results to root/project/fullname.")
@click.option("--project",    type=click.Path(exists=False), default=None,                  help="Save results to root/project/fullname.")
@click.option("--fullname",   type=str,                      default=None,                  help="Save results to root/project/fullname.")
@click.option("--weights",    type=click.Path(exists=False), default=None,                  help="Weights paths.")
@click.option("--batch-size", type=int,                      default=1,                     help="Total Batch size for all GPUs.")
@click.option("--image-size", type=int,                      default=512,                   help="Image sizes.")
@click.option("--devices",    type=str,                      default=None,                  help="Running devices.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
@click.pass_context
def main(
    ctx,
    config     : str,
    input_dir  : mon.Path | str,
    output_dir : mon.Path | str,
    name       : str,
    variant    : int | str | None,
    data       : str,
    root       : mon.Path | str,
    project    : str,
    fullname   : str | None,
    weights    : Any,
    batch_size : int,
    image_size : int | list[int],
    seed       : int,
    devices    : str,
    resize     : bool,
    benchmark  : bool,
    save_image : bool,
    verbose    : bool
):
    model_kwargs = {
        k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--"))
            else True for i, k in enumerate(ctx.args) if k.startswith("--")
    }
    
    # Obtain arguments
    hostname      = socket.gethostname().lower()
    config_module = mon.get_config_module(
        project = project,
        name    = name,
        variant = variant,
        data    = data,
        config  = config,
    )
    config_args = importlib.import_module(f"{config_module}")
    
    # Prioritize input args --> config file args
    input_dir   = mon.Path(input_dir)
    project     = project or config_args.model["project"]
    project     = str(project).replace(".", "/")
    root        = root
    fullname    = fullname    or mon.get_model_fullname(name, data, variant) or config_args.model["fullname"]
    variant     = variant     or config_args.model["variant"]
    variant     = None if variant in ["", "none", "None"] else variant
    weights     = weights     or config_args.model["weights"]
    batch_size  = batch_size  or config_args.datamodule["batch_size"]
    image_size  = image_size  or config_args.datamodule["image_size"]
    seed        = seed        or config_args["seed"] or random.randint(1, 10000)
    devices     = devices     or config_args.trainer["devices"]

    # Update arguments
    args                 = mon.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["root"]         = mon.Path(root)
    args["project"]      = project
    args["fullname"]     = fullname
    args["image_size"]   = image_size
    args["seed"]         = seed
    args["verbose"]      = verbose
    args["config_file"]  = config_args.__file__,
    args["datamodule"]  |= {
        "root"      : input_dir,
        "resize"    : resize,
        "image_size": image_size,
        "batch_size": batch_size,
        # "verbose"   : verbose,
    }
    args["model"] |= {
        "weights"  : weights,
        "name"     : fullname,
        "variant"  : variant,
        "root"     : mon.Path(root),
        "project"  : project,
        # "verbose"  : verbose,
    }
    args["model"]     |= model_kwargs
    args["predictor"] |= {
        "devices"   : devices,
        "benchmark" : benchmark,
        "output_dir": mon.Path(output_dir),
        "save_image": save_image,
        "verbose"   : verbose,
    }
    predict(args=args)

# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
