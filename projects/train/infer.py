#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script.
"""

from __future__ import annotations

import argparse
import socket
import one.vision

from one.data import *
from one.nn import attempt_load
from one.nn import ImageInferrer


# H1: - Infer ------------------------------------------------------------------

def infer(args: Munch | dict):
    args = Munch.fromDict(args)
    
    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    # Model
    devices = select_device(device=args.devices)
    model   = attempt_load(
        name        = args.model,
        cfg         = args.cfg,
        weights     = args.weights,
        num_classes = args.num_classes,
        phase       = "inference",
    )
    print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # H2: - Inferrer -----------------------------------------------------------
    console.rule("[bold red]2. Inferrer")
    inferrer = ImageInferrer(
        source     = args.source,
        root       = args.root,
        name       = args.name,
        batch_size = args.batch_size,
        shape      = args.shape,
        device     = args.devices,
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
        "model"      : "zerodce++",
        "cfg"        : "zerodce++.yaml",
        "weights"    : PRETRAINED_DIR / "zerodce++" / "zerodce++-lol.pt",
        "num_classes": None,
        "source"     : DATA_DIR / "lol226",
        "batch_size" : 1,
        "img_size"   : (3, 256, 256),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "lp-labdesktop02-ubuntu": {
        "model"      : "zerodce++",     
        "cfg"        : "zerodce++.yaml",
        "weights"    : None,
        "num_classes": None,
        "source"     : DATA_DIR / "lol226",
        "batch_size" : 1,
        "img_size"   : (3, 256, 256),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "lp-imac.local": {
        "model"      : "zerodce",
        "cfg"        : "zerodce.yaml",
        "weights"    : PRETRAINED_DIR / "zerodce" / "zerodce-lol226.pt",
        "num_classes": None,
        "source"     : DATA_DIR / "lol226",  # / "test" / "low",
        "batch_size" : 1,
        "img_size"   : (3, 256, 256),
		"devices"    : "cpu",
        "root"       : RUNS_DIR / "infer",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,                             help="Model name")
    parser.add_argument("--cfg",         type=str,                             help="Model config.")
    parser.add_argument("--weights",     type=str,                             help="Weights path.")
    parser.add_argument("--num-classes", type=int,                             help="Number of classes.")
    parser.add_argument("--source",      type=str,                             help="Data source.")
    parser.add_argument("--batch-size",  type=int,                             help="Total Batch size for all GPUs.")
    parser.add_argument("--img-size",    type=int,  nargs="+",                 help="Image sizes.")
    parser.add_argument("--devices",     type=str,                             help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--root",        type=str, default=RUNS_DIR / "infer", help="Save results to root/name")
    parser.add_argument("--name",        type=str,                             help="Save results to root/name")
    parser.add_argument("--save",        action="store_true", default=True,    help="Save.")
    parser.add_argument("--verbose",     action="store_true", default=True,    help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    hostname    = socket.gethostname().lower()
    host_args   = Munch(hosts[hostname])
    
    input_args  = vars(parse_args())
    model       = input_args.get("model",       None) or host_args.get("model",       None)
    cfg         = input_args.get("cfg",         None) or host_args.get("cfg",         None)
    weights     = input_args.get("weights",     None) or host_args.get("weights",     None)
    num_classes = input_args.get("num_classes", None) or host_args.get("num_classes", None)
    source      = input_args.get("source",      None) or host_args.get("source",      None)
    batch_size  = input_args.get("batch_size",  None) or host_args.get("batch_size",  None)
    shape       = input_args.get("img_size",    None) or host_args.get("img_size",    None)
    devices     = input_args.get("devices",     None) or host_args.get("devices",     None)
    root        = input_args.get("root",        None) or host_args.get("root",        None)
    name        = input_args.get("name",        None) or host_args.get("name",        None)
    save        = input_args.get("save",        None) or host_args.get("save",        None)
    verbose     = input_args.get("verbose",     None) or host_args.get("verbose",     None)
    
    args = Munch(
        hostname    = hostname,
        model       = model,
        cfg         = cfg,
        weights     = weights,
        num_classes = num_classes,
        source      = source,
        batch_size  = batch_size,
        shape       = shape,
        devices     = devices,
        root        = root,
        name        = name,
        save        = save,
        verbose     = verbose,
    )
    infer(args)
