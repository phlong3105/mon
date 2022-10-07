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
from one.nn import VisionInferrer


# H1: - Infer ------------------------------------------------------------------

def infer(args: Munch | dict):
    args = Munch.fromDict(args)
    
    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    # Model
    model = attempt_load(
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
    inferrer = VisionInferrer(
        source      = args.source,
        project     = args.project,
        root        = args.root,
        name        = args.name,
        max_samples = args.max_samples,
        batch_size  = args.batch_size,
        shape       = args.shape,
        device      = args.devices,
        phase       = model.phase,
        save        = args.save,
        verbose     = args.verbose,
    )
    console.log("[green]Done")
    
    # H2: - Inference ----------------------------------------------------------
    console.rule("[bold red]3. INFERENCE")
    inferrer.run(model=model, source=args.source)
    console.log("[green]Done")


# H1: - Main -------------------------------------------------------------------

hosts = {
	"lp-labdesktop01-ubuntu": {
        "model"      : "zerodcev2",
        "cfg"        : "zerodcev2-s5",
        "weights"    : PRETRAINED_DIR / "zerodcev2" / "zerodcev2-s5-lol4k.pt",
        "num_classes": None,
        "source"     : DATA_DIR / "lol" / "demo" / "landscapes.mp4",  # / "train" / "low",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,  # (3, 900, 1200),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "project"    : "lol4k",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "lp-labdesktop02-ubuntu": {
        "model"      : "zerodcev2",
        "cfg"        : "zerodcev2-s5",
        "weights"    : PRETRAINED_DIR / "zerodcev2" / "zerodcev2-s5-lol226.pt",
        "num_classes": None,
        "source"     : DATA_DIR / "lol_demo" / "aokigahara.mp4",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,  # (3, 256, 256),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "project"    : "lol_demo",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "vsw-ws02": {
        "model"      : "hinet-derain",
        "cfg"        : "hinet.yaml",
        "weights"    : PRETRAINED_DIR / "hinet" / "hinet-derain-cityscapes_rain.pt",
        "num_classes": None,
        "source"     : DATA_DIR / "cityscapes" / "leftImg8bit_rain",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,  # (3, 512, 512),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "project"    : None,
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "vsw-ws03": {
        "model"      : "zerodcev2",
        "cfg"        : "zerodcev2-s5",
        "weights"    : PRETRAINED_DIR / "zerodcev2" / "zerodcev2-s5-lol4k.pt",
        "num_classes": None,
        "source"     : DATA_DIR / "lol" / "train" / "low",
        "max_samples": None,
        "batch_size" : 1,
        "img_size"   : None,  # (3, 900, 1200),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "project"    : "lol4k",
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
    parser.add_argument("--max-samples", type=int,                             help="Only process certain amount of samples.")
    parser.add_argument("--batch-size",  type=int,                             help="Total Batch size for all GPUs.")
    parser.add_argument("--img-size",    type=int, nargs="+",                  help="Image sizes.")
    parser.add_argument("--devices",     type=str,                             help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--root",        type=str, default=RUNS_DIR / "infer", help="Save results to root/project/name")
    parser.add_argument("--project",     type=str,                             help="Save results to root/project/name")
    parser.add_argument("--name",        type=str,                             help="Save results to root/project/name")
    parser.add_argument("--save",        action="store_true",                  help="Save.")
    parser.add_argument("--verbose",     action="store_true",                  help="Display results.")
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
    max_samples = input_args.get("max_samples", None) or host_args.get("max_samples", None)
    batch_size  = input_args.get("batch_size",  None) or host_args.get("batch_size",  None)
    shape       = input_args.get("img_size",    None) or host_args.get("img_size",    None)
    devices     = input_args.get("devices",     None) or host_args.get("devices",     None)
    root        = input_args.get("root",        None) or host_args.get("root",        None)
    project     = input_args.get("project",     None) or host_args.get("project",     None)
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
        max_samples = max_samples,
        batch_size  = batch_size,
        shape       = shape,
        devices     = devices,
        root        = root,
        name        = name,
        project     = project,
        save        = save,
        verbose     = verbose,
    )
    infer(args)
