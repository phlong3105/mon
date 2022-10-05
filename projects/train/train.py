#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script.
"""

from __future__ import annotations

import argparse
import socket

from pytorch_lightning.callbacks import Checkpoint

from one.data import *
from one.datasets import *
from one.nn import BaseModel
from one.nn import get_epoch
from one.nn import get_global_step
from one.nn import get_latest_checkpoint
from one.nn import TensorBoardLogger
from one.nn import Trainer


# H1: - Train ------------------------------------------------------------------

def train(args: Munch | dict):
    args = Munch.fromDict(args)
    
    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    
    data: DataModule = DATAMODULES.build_from_dict(cfg=args.data)
    data.prepare_data()
    data.setup(phase="training")
    
    args.model.classlabels = data.classlabels
    model: BaseModel       = MODELS.build_from_dict(cfg=args.model)
    model.phase            = "training"
    
    print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # H2: - Trainer ------------------------------------------------------------
    console.rule("[bold red]2. SETUP TRAINER")
    copy_file_to(file=args.cfg_file, dst=model.root)
    
    ckpt                 = get_latest_checkpoint(dirpath=model.weights_dir)
    callbacks            = CALLBACKS.build_from_dictlist(cfgs=args.callbacks)
    enable_checkpointing = any(isinstance(cb, Checkpoint) for cb in callbacks)
    
    logger = []
    for k, v in args.logger.items():
        if k == "tensorboard":
            v |= { "save_dir": model.root }
            logger.append(TensorBoardLogger(**v))
    
    args.trainer.callbacks            = callbacks
    args.trainer.default_root_dir     = model.root
    args.trainer.enable_checkpointing = enable_checkpointing
    args.trainer.logger               = logger
    args.trainer.num_sanity_val_steps = (0 if (ckpt is not None) else args.trainer.num_sanity_val_steps)
    
    trainer               = Trainer(**args.trainer)
    trainer.current_epoch = get_epoch(ckpt=ckpt)
    trainer.global_step   = get_global_step(ckpt=ckpt)
    console.log("[green]Done")
    
    # H2: - Training -----------------------------------------------------------
    console.rule("[bold red]3. TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = data.train_dataloader,
        val_dataloaders   = data.val_dataloader,
        ckpt_path         = ckpt,
    )
    console.log("[green]Done")


# H1: - Main -------------------------------------------------------------------

hosts = {
	"lp-labdesktop01-ubuntu": {
		"cfg"        : "zerodcepp_lol4k",
        "project"    : "lol4k",
        "weights"    : None,
        "batch_size" : 8,
        "accelerator": "auto",
		"devices"    : 1,
        "max_epochs" : 500,
		"strategy"   : None,
	},
    "lp-labdesktop02-ubuntu": {
		"cfg"        : "zerodcev2_s1_lol4k",
        "project"    : "lol4k",
        "weights"    : None,
        "batch_size" : 4,
        "accelerator": "auto",
		"devices"    : 1,
        "max_epochs" : 500,
		"strategy"   : None,
	},
    "vsw-ws02": {
		"cfg"        : "hinet_derain_cityscapes_rain",
        "project"    : "hinet",
        "weights"    : None,
        "batch_size" : 8,
        "accelerator": "auto",
		"devices"    : 1,
        "max_epochs" : 200,
		"strategy"   : None,
	},
    "vsw-ws03": {
		"cfg"        : "zerodcev2_s5_lol4k",
        "project"    : "lol4k",
        "weights"    : None,
        "batch_size" : 16,
        "accelerator": "auto",
		"devices"    : 1,
        "max_epochs" : 500,
		"strategy"   : None,
	},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",         type=str,            help="The training cfg to use.")
    parser.add_argument("--project",     type=str,            help="Save results to root/project/name")
    parser.add_argument("--weights",     type=str,            help="Weights path.")
    parser.add_argument("--batch-size",  type=int,            help="Total Batch size for all GPUs.")
    parser.add_argument("--img-size",    type=int, nargs="+", help="Image sizes.")
    parser.add_argument("--accelerator", type=str,            help="Supports passing different accelerator types ('cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto') as well as custom accelerator instances.")
    parser.add_argument("--devices",     type=str,            help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--max-epochs",  type=int,            help="Stop training once this number of epochs is reached.")
    parser.add_argument("--strategy",    type=str,            help="Supports different training strategies with aliases as well custom strategies.")
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
    
    project     = input_args.get("project",     None) or host_args.get("project",     None) or module.model["project"]
    weights     = input_args.get("weights",     None) or host_args.get("weights",     None) or module.model["pretrained"]
    batch_size  = input_args.get("batch_size",  None) or host_args.get("batch_size",  None) or module.data["batch_size"]
    shape       = input_args.get("img_size",    None) or host_args.get("img_size",    None) or module.data["shape"]
    accelerator = input_args.get("accelerator", None) or host_args.get("accelerator", None) or module.trainer["accelerator"]
    devices     = input_args.get("devices",     None) or host_args.get("devices",     None) or module.trainer["devices"]
    max_epochs  = input_args.get("max_epochs",  None) or host_args.get("max_epochs",  None) or module.trainer["max_epochs"]
    strategy    = input_args.get("strategy",    None) or host_args.get("strategy",    None) or module.trainer["strategy"]
    
    args = Munch(
        hostname  = hostname,
        cfg_file  = module.__file__,
        data      = module.data | {
            "shape"     : shape,
            "batch_size": batch_size,
        },
        model     = module.model | {
            "project"   : project,
            "pretrained": weights,
        },
        callbacks = module.callbacks,
        logger    = module.logger,
        trainer   = module.trainer | {
            "accelerator": accelerator,
            "devices"    : devices,
            "max_epochs" : max_epochs,
            "strategy"   : strategy,
        },
    )
    train(args)
