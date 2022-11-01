#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script.
"""

from __future__ import annotations

import argparse
import importlib
import socket
import one.vision

from one.data import *
from one.nn import VisionInferrer


# H1: - Infer ------------------------------------------------------------------

def infer(args: Munch | dict):
    args = Munch.fromDict(args)
    
    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    # Model
    model       = MODELS.build_from_dict(cfg=args.model)
    model.phase = "training"
    print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # H2: - Inferrer -----------------------------------------------------------
    console.rule("[bold red]2. Inferrer")
    inferrer = VisionInferrer(
        source     = args.source,
        project    = args.project,
        root       = args.root,
        name       = args.name,
        max_samples= args.max_samples,
        batch_size = args.batch_size,
        shape      = args.shape,
        device     = args.devices,
        phase      = model.phase,
        save       = args.save,
        verbose    = args.verbose,
    )
    console.log("[green]Done")
    
    # H2: - Inference ----------------------------------------------------------
    console.rule("[bold red]3. INFERENCE")
    inferrer.run(model=model, source=args.source)
    console.log("[green]Done")


# H1: - Main -------------------------------------------------------------------

hosts = {
	"lp-labdesktop01-ubuntu": {
        "model"      : "zeroadce",
        "cfg"        : "zeroadce_e_lol226",
        # "weights"    : None,
        "weights"    : PROJECTS_DIR / "train" / "runs" / "train" / "lol226" / "zeroadce-e-lol226" / "weights" / "best.pt",
        "root"       : RUNS_DIR / "infer",
        "project"    : "lol226",
        "name"       : "landscapes",
        "num_classes": None,
        "source"     : DATA_DIR / "lol" / "demo" / "landscapes.mp4",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,
		"devices"    : "0",
        "save"       : True,
        "verbose"    : False,
	},
    "vsw-ws02": {
        "model"      : "zeroadce",
        "cfg"        : "zeroadce_a_lol226",
        # "weights"    : None,
        "weights"    : PROJECTS_DIR / "train" / "runs" / "train" / "lol226" / "zeroadce-a-lol226" / "weights" / "best.pt",
        "root"       : RUNS_DIR / "infer",
        "project"    : "lol226",
        "name"       : "landscapes",
        "num_classes": None,
        "source"     : DATA_DIR / "lol" / "demo" / "landscapes.mp4",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,
		"devices"    : "0",
        "save"       : True,
        "verbose"    : True,
	},
    "vsw-ws03": {
        "model"      : "zerodce",
        "cfg"        : "zerodce_lol_demo",
        "weights"    : PRETRAINED_DIR / "zerodce" / "zerodce-lol226.pt",
        "root"       : RUNS_DIR / "infer",
        "project"    : "lol_demo",
        "name"       : "exp",
        "num_classes": None,
        "source"     : DATA_DIR / "lol_demo" / "truck.mp4",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,
		"devices"    : "0",
        "save"       : True,
        "verbose"    : True,
	},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,                             help="Model name")
    parser.add_argument("--cfg",         type=str,                             help="Model config.")
    parser.add_argument("--weights",     type=str,                             help="Weights path.")
    parser.add_argument("--root",        type=str, default=RUNS_DIR / "infer", help="Save results to root/project/name")
    parser.add_argument("--project",     type=str,                             help="Save results to root/project/name")
    parser.add_argument("--name",        type=str,                             help="Save results to root/project/name")
    parser.add_argument("--num-classes", type=int,                             help="Number of classes.")
    parser.add_argument("--source",      type=str,                             help="Data source.")
    parser.add_argument("--max-samples", type=int,                             help="Only process certain amount of samples.")
    parser.add_argument("--batch-size",  type=int,                             help="Total Batch size for all GPUs.")
    parser.add_argument("--img-size",    type=int, nargs="+",                  help="Image sizes.")
    parser.add_argument("--devices",     type=str,                             help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--save",        action="store_true",                  help="Save.")
    parser.add_argument("--verbose",     action="store_true",                  help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    hostname    = socket.gethostname().lower()
    host_args   = Munch(hosts[hostname])
    input_args  = vars(parse_args())
    cfg         = input_args.get("cfg",     None) or host_args.get("cfg",     None)
    project     = input_args.get("project", None) or host_args.get("project", None)
    
    if project is not None and project != "":
        module = importlib.import_module(f"one.cfg.{project}.{cfg}")
    else:
        module = importlib.import_module(f"one.cfg.{cfg}")
        
    # model       = input_args.get("model",       None) or host_args.get("model",       None)
    cfg         = input_args.get("cfg",         None) or host_args.get("cfg",         None)
    weights     = input_args.get("weights",     None) or host_args.get("weights",     None)
    root        = input_args.get("root",        None) or host_args.get("root",        None)
    project     = input_args.get("project",     None) or host_args.get("project",     None)
    name        = input_args.get("name",        None) or host_args.get("name",        None)
    num_classes = input_args.get("num_classes", None) or host_args.get("num_classes", None)
    source      = input_args.get("source",      None) or host_args.get("source",      None)
    max_samples = input_args.get("max_samples", None) or host_args.get("max_samples", None)
    batch_size  = input_args.get("batch_size",  None) or host_args.get("batch_size",  None)
    shape       = input_args.get("img_size",    None) or host_args.get("img_size",    None)
    devices     = input_args.get("devices",     None) or host_args.get("devices",     None)
    save        = input_args.get("save",        None) or host_args.get("save",        None)
    verbose     = input_args.get("verbose",     None) or host_args.get("verbose",     None)
    
    args = Munch(
        hostname    = hostname,
        model       = module.model | {
            "project"   : project,
            "pretrained": weights,
        },
        cfg         = cfg,
        weights     = weights,
        root        = root,
        project     = project,
        name        = name,
        num_classes = num_classes,
        source      = source,
        max_samples = max_samples,
        batch_size  = batch_size,
        shape       = shape,
        devices     = devices,
        save        = save,
        verbose     = verbose,
    )
    infer(args)
