#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prediction pipeline."""

from __future__ import annotations

import importlib
import socket
import time
from typing import Any

import click
import cv2
import numpy as np
import torch
import torchvision

import mon

console = mon.console


# region Host

hosts = {
    "lp-macbookpro.local": {
        "config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : None,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
    },
	"lp-labdesktop-01": {
		"config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
    "vsw-ws02": {
		"config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
    "vsw-ws-03": {
		"config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
}

# endregion


# region Function

def predict(args: dict):
    # Initialization
    model_name    = args["model"]["name"]
    variant       = args["model"]["variant"]
    variant       = variant if variant not in [None, "", "none"] else None
    model_variant = f"{model_name}-{variant}" if variant is not None else f"{model_name}"
    console.rule(f"[bold red] {model_variant}")
    
    weights = args["model"]["weights"]
    model: mon.Model = mon.MODELS.build(config=args["model"])
    if torch.cuda.is_available():
        devices = torch.device(f"cuda:0")
    else:
        devices = torch.device("cpu")
    state_dict  = torch.load(weights, map_location=devices)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    model.phase = mon.ModelPhase.INFERENCE
    model.eval()

    output_dir = args["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Measure efficiency score
    if args["benchmark"] and torch.cuda.is_available():
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = model,
            image_size = args["image_size"],
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
     
    # Data
    data       = mon.Path(args["datamodule"]["root"])
    image_size = args["datamodule"]["image_size"]
    h, w       = mon.get_hw(image_size)
    resize     = args["datamodule"]["resize"]
    console.log(f"{data}")
    
    if data.is_video_file():
        image_loader = mon.VideoLoaderCV(source=data, to_rgb=True, to_tensor=True, normalize=True)
        video_writer = mon.VideoWriterCV(
            destination = output_dir / data.stem,
            image_size  = [480, 640],
            frame_rate  = 30,
            fourcc      = "mp4v",
            save_image  = False,
            denormalize = True,
            verbose     = False,
        )
    else:
        image_loader = mon.ImageLoader(source=data, to_rgb=True, to_tensor=True, normalize=True)
        video_writer = None

    #
    with torch.no_grad():
        # image_paths = list(data.rglob("*"))
        # image_paths = [path for path in image_paths if path.is_image_file()]
        # image_paths.sort()
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, indexes, files, rel_paths in pbar.track(
                sequence    = image_loader,
                total       = len(image_loader),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                # image       = mon.read_image(path=image_path, to_rgb=True, to_tensor=True, normalize=True)
                if resize:
                    h0, w0  = mon.get_image_size(images)
                    images  = mon.resize(input=images, size=[h, w])
                input       = images.to(model.device)
                start_time  = time.time()
                output      = model(input=input, augment=False, profile=False, out_index=-1)
                '''
                output       = model(input=input, augment=False, profile=False)
                a, p, output = output[0], output[1], output[2]
                a = (-1 * a)
                a = mon.to_image_nparray(a, False, True)
                p = mon.to_image_nparray(p, False, True)
                (B, G, R) = cv2.split(a)
                zeros = np.zeros(a.shape[:2], dtype=a.dtype)
                R = cv2.merge([zeros, zeros, R])
                G = cv2.merge([zeros, G, zeros])
                B = cv2.merge([B, zeros, zeros])
                '''
                run_time    = time.time() - start_time
                output      = output[-1] if isinstance(output, (list, tuple)) else output
                if resize:
                    output  = mon.resize(input=images, size=[h0, w0])

                if args["save_image"]:
                    result_path = output_dir / f"{files[0].stem}.png"
                    torchvision.utils.save_image(output, str(result_path))
                    '''
                    a_path = output_dir / f"{files[0].stem}-a.png"
                    B_path = output_dir / f"{files[0].stem}-b.png"
                    G_path = output_dir / f"{files[0].stem}-g.png"
                    R_path = output_dir / f"{files[0].stem}-r.png"
                    p_path = output_dir / f"{files[0].stem}-p.png"
                    cv2.imwrite(str(a_path), a)
                    cv2.imwrite(str(B_path), B)
                    cv2.imwrite(str(G_path), G)
                    cv2.imwrite(str(R_path), R)
                    cv2.imwrite(str(p_path), p)
                    '''
                    if data.is_video_file():
                        video_writer.write_batch(images=output)
                sum_time += run_time
        avg_time = float(sum_time / len(image_loader))
        console.log(f"Average time: {avg_time}")


@click.command(context_settings=dict(
    ignore_unknown_options = True,
    allow_extra_args       = True,
))
@click.option("--data",        default=mon.DATA_DIR,          type=click.Path(exists=True),  help="Source data directory.")
@click.option("--config",      default="",                    type=click.Path(exists=False), help="The training config to use.")
@click.option("--root",        default=mon.RUN_DIR/"predict", type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--project",     default=None,                  type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--name",        default=None,                  type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--variant",     default=None,                  type=str,                      help="Model variant.")
@click.option("--weights",     default=None,                  type=click.Path(exists=False), help="Weights paths.")
@click.option("--batch-size",  default=1,                     type=int,                      help="Total Batch size for all GPUs.")
@click.option("--image-size",  default=512,                   type=int,                      help="Image sizes.")
@click.option("--resize",      is_flag=True)
@click.option("--benchmark",   is_flag=True)
@click.option("--output-dir",  default=mon.RUN_DIR/"predict", type=click.Path(exists=False), help="Save results location.")
@click.option("--save-image",  is_flag=True)
@click.option("--verbose",     is_flag=True)
@click.pass_context
def main(
    ctx,
    data       : mon.Path | str,
    config     : mon.Path | str,
    root       : mon.Path | str,
    project    : str,
    name       : str,
    variant    : int | str | None,
    weights    : Any,
    batch_size : int,
    image_size : int | list[int],
    resize     : bool,
    benchmark  : bool,
    output_dir : mon.Path | str,
    save_image : bool,
    verbose    : bool
):
    model_kwargs = {
        k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--"))
            else True for i, k in enumerate(ctx.args) if k.startswith("--")
    }

    # Obtain arguments
    hostname  = socket.gethostname().lower()
    host_args = hosts[hostname]
    config    = config  or host_args.get("config",  None)
    project   = project or host_args.get("project", None)
    
    if project is not None and project != "":
        project_module = project.replace("/", ".")
        config_args    = importlib.import_module(f"config.{project_module}.{config}")
    else:
        config_args    = importlib.import_module(f"config.{config}")
    
    # Prioritize input args --> predefined args --> config file args
    data        = mon.Path(data)
    project     = project or config_args.model["project"]
    project     = str(project).replace(".", "/")
    root        = root        or host_args.get("root",       None)
    name        = name        or host_args.get("name",       None) or config_args.model["name"]
    variant     = variant     or host_args.get("variant",    None) or config_args.model["variant"]
    variant     = None if variant in ["", "none", "None"] else variant
    weights     = weights     or host_args.get("weights",    None) or config_args.model["weights"]
    batch_size  = batch_size  or host_args.get("batch_size", None) or config_args.data["batch_size"]
    image_size  = image_size  or host_args.get("image_size", None) or config_args.data["image_size"]
    
    # Update arguments
    args                 = mon.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["root"]         = mon.Path(root)
    args["project"]      = project
    args["image_size"]   = image_size
    args["output_dir"]   = mon.Path(output_dir)
    args["config_file"]  = config_args.__file__,
    args["datamodule"]  |= {
        "root"      : data,
        "resize"    : resize,
        "image_size": image_size,
        "batch_size": batch_size,
    }
    args["model"] |= {
        "weights": weights,
        "name"   : name,
        "variant": variant,
        "root"   : root,
        "project": project,
        "verbose": verbose,
    }
    args["model"]      |= model_kwargs
    args["save_image"]  = save_image
    args["benchmark"]   = benchmark
    predict(args=args)

# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
