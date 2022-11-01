#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script.
"""

from __future__ import annotations

import argparse
import socket

from torch.backends import cudnn

import one.vision

from one.data import *
from one.nn import attempt_load
from one.nn import VisionInferrer

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR    = CURRENT_DIR / "data"
RUNS_DIR    = CURRENT_DIR / "runs"
WEIGHTS_DIR = CURRENT_DIR / "weights"


# H1: - Infer ------------------------------------------------------------------

def infer(args: Munch | dict):
    args = Munch.fromDict(args)
    
    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    # Model
    model_fullname = str(args.weights.stem)
    model          = attempt_load(
        name        = args.model,
        cfg         = args.cfg,
        weights     = args.weights,
        fullname    = model_fullname,
        num_classes = args.num_classes,
        phase       = "inference",
    )
    # CuDNN
    cudnn.benchmark = True
    cudnn.enabled   = True
    print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # H2: - Inferrer -----------------------------------------------------------
    console.rule("[bold red]2. INFERRER")
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
        tensorrt    = args.tensorrt,
        save        = args.save,
        verbose     = args.verbose,
    )
    console.log("[green]Done")
    
    # H2: - Inference ----------------------------------------------------------
    console.rule("[bold red]3. INFERENCE")
    inferrer.run(model=model, source=args.source)
    console.log("[green]Done")


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,            default="zeroadce",              help="Model name")
    parser.add_argument("--cfg",         type=str,            default="zeroadce",              help="Model config.")
    parser.add_argument("--weights",     type=str,            default=WEIGHTS_DIR / "best.pt", help="Weights path.")
    parser.add_argument("--root",        type=str,            default=RUNS_DIR / "infer",      help="Save results to root/project/name")
    parser.add_argument("--project",     type=str,            default="test",                  help="Save results to root/project/name")
    parser.add_argument("--name",        type=str,            default="zeroadce",              help="Save results to root/project/name")
    parser.add_argument("--num-classes", type=int,            default=None,                    help="Number of classes.")
    parser.add_argument("--source",      type=str,            default=DATA_DIR / "test",       help="Data source.")
    parser.add_argument("--max-samples", type=int,            default=None,                    help="Only process certain amount of samples.")
    parser.add_argument("--batch-size",  type=int,            default=1,                       help="Total Batch size for all GPUs.")
    parser.add_argument("--img-size",    type=int, nargs="+", default=None,                    help="Image sizes.")
    parser.add_argument("--devices",     type=str,            default="0",                     help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--tensorrt",    action="store_true", default=False,                   help="Use TensorRT.")
    parser.add_argument("--save",        action="store_true", default=True,                    help="Save.")
    parser.add_argument("--verbose",     action="store_true", default=True,                    help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    hostname    = socket.gethostname().lower()
    input_args  = vars(parse_args())
    model       = input_args.get("model",       None)
    cfg         = input_args.get("cfg",         None)
    weights     = input_args.get("weights",     None)
    root        = input_args.get("root",        None)
    project     = input_args.get("project",     None)
    name        = input_args.get("name",        None)
    num_classes = input_args.get("num_classes", None)
    source      = input_args.get("source",      None)
    max_samples = input_args.get("max_samples", None)
    batch_size  = input_args.get("batch_size",  None)
    shape       = input_args.get("img_size",    None)
    devices     = input_args.get("devices",     None)
    tensorrt    = input_args.get("tensorrt",    None)
    save        = input_args.get("save",        None)
    verbose     = input_args.get("verbose",     None)

    args = Munch(
        hostname    = hostname,
        model       = model,
        cfg         = cfg,
        weights     = weights,
        root        = root,
        name        = name,
        project     = project,
        num_classes = num_classes,
        source      = source,
        max_samples = max_samples,
        batch_size  = batch_size,
        shape       = shape,
        devices     = devices,
        tensorrt    = tensorrt,
        save        = save,
        verbose     = verbose,
    )
    infer(args=args)
