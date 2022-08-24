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
    
    args.model.classlabels = data.classlabels
    model: BaseModel       = MODELS.build_from_dict(cfg=args.model)

    print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # H2: - Trainer ------------------------------------------------------------
    console.rule("[bold red]2. SETUP TRAINER")
    copy_file_to(file=args.cfg_file, dst=model.root)
    data.setup(phase="training")
    model.phase = "training"
    
    ckpt                 = get_latest_checkpoint(dirpath=model.weights_dir)
    callbacks            = CALLBACKS.build_from_dictlist(cfgs=args.callbacks)
    enable_checkpointing = any(isinstance(cb, Checkpoint) for cb in callbacks)
    
    logger = []
    for k, v in args.logger.items():
        if k == "tensorboard":
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
		"cfg"        : "zerodce_lol226",
        "accelerator": "auto",
		"devices"    :  1,
		"strategy"   : None,
	},
    "lp-imac.local": {
		"cfg"        : "zerodce_lol226",
        "accelerator": "cpu",
		"devices"    :  1,
		"strategy"   : None,
	},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",         type=str, help="The training cfg to use.")
    parser.add_argument("--accelerator", type=str, help="Supports passing different accelerator types ('cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto') as well as custom accelerator instances.")
    parser.add_argument("--devices",     type=str, help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--strategy",    type=str, help="Supports different training strategies with aliases as well custom strategies.")

    args   = parser.parse_args()
    return args


if __name__ == "__main__":
    hostname    = socket.gethostname().lower()
    args        = Munch(hosts[hostname])
    
    input_args  = vars(parse_args())
    cfg         = input_args.get("cfg",         None) or args.cfg
    accelerator = input_args.get("accelerator", None) or args.accelerator
    devices     = input_args.get("devices"    , None) or args.devices
    strategy    = input_args.get("strategy"   , None) or args.strategy
    
    module = importlib.import_module(f"one.cfg.{args.cfg}")
    args   = Munch(
        hostname  = hostname,
        cfg_file  = module.__file__,
        data      = module.data,
        model     = module.model,
        callbacks = module.callbacks,
        logger    = module.logger,
        trainer   = module.trainer | {
            "accelerator": accelerator,
            "devices"    : devices,
            "strategy"   : strategy,
        },
    )
    
    train(args)
