#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements training pipeline."""

from __future__ import annotations

import importlib
import socket
from typing import Any

import click
from lightning.pytorch import callbacks as lcallbacks

from mon import core as mf, nn
from mon.globals import CALLBACKS, DATAMODULES, MODELS, RUN_DIR

console = mf.console

# region Host

hosts = {
	"lp-labdesktop-01-ubuntu": {
        "config"     : "zerodcev2_lol226",
        "root"       : RUN_DIR/"train",
        "project"    : "zerodcev2",
        "name"       : "zerodcev2-lol226",
        "weights"    : None,
        "batch_size" : 4,
        "image_size" : (512, 512),
        "accelerator": "auto",
        "devices"    : 1,
        "max_epochs" : None,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
	},
    "lp-labdesktop-02-ubuntu": {
        "config"     : "zerodcev2_lol226",
        "root"       : RUN_DIR/"train",
        "project"    : "zerodcev2",
        "name"       : "zerodcev2-lol226",
        "weights"    : None,
        "batch_size" : 4,
        "image_size" : (512, 512),
        "accelerator": "auto",
        "devices"    : 1,
        "max_epochs" : None,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
	},
    "vsw-ws01": {
        "config"     : "hinet_gt_rain",
        "root"       : RUN_DIR/"train",
        "project"    : "hinet",
        "name"       : "hinet-gt-rain",
        "weights"    : None,
        "batch_size" : 4,
        "image_size" : (512, 512),
        "accelerator": "auto",
        "devices"    : 1,
        "max_epochs" : None,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
	},
    "vsw-ws02": {
        "config"     : "",
        "root"       : RUN_DIR/"train",
        "project"    : None,
        "name"       : None,
        "weights"    : None,
        "batch_size" : 4,
        "image_size" : (256, 256),
        "accelerator": "auto",
        "devices"    : 1,
        "max_epochs" : None,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
	},
    "vsw-ws03": {
        "config"     : "",
        "root"       : RUN_DIR/"train",
        "project"    : None,
        "name"       : None,
        "weights"    : None,
        "batch_size" : 4,
        "image_size" : (256, 256),
        "accelerator": "auto",
        "devices"    : 1,
        "max_epochs" : None,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
	},
}

# endregion


# region Function

@click.command()
@click.option("--config",      default="",     type=click.Path(exists=False), help="The training config to use.")
@click.option("--root",        default=RUN_DIR/"train", type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--project",     default=None,   type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--name",        default=None,   type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--weights",     default=None,   type=click.Path(exists=False), help="Weights paths.")
@click.option("--batch-size",  default=8,      type=int,  help="Total Batch size for all GPUs.")
@click.option("--image-size",  default=None,   type=int,  help="Image sizes.")
@click.option("--accelerator", default="gpu",  type=click.Choice(["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"], case_sensitive=False))
@click.option("--devices",     default=0,      type=int,  help="Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`.")
@click.option("--max-epochs",  default=100,    type=int,  help="Stop training once this number of epochs is reached.")
@click.option("--max-steps",   default=None,   type=int,  help="Stop training once this number of steps is reached.")
@click.option("--strategy",    default="auto", type=str,  help="Supports different training strategies with aliases as well custom strategies.")
@click.option("--exist-ok",    is_flag=True,   help="Whether to overwrite existing experiment.")
def train(
    config     : mf.Path | str,
    root       : mf.Path,
    project    : str,
    name       : str,
    weights    : Any,
    batch_size : int,
    image_size : int | list[int],
    accelerator: str,
    devices    : int | str | list[int, str],
    max_epochs : int,
    max_steps  : int,
    strategy   : str,
    exist_ok   : bool,
):
    # Obtain arguments
    hostname  = socket.gethostname().lower()
    host_args = hosts[hostname]
    config    = config  or host_args.get("config",  None)
    project   = project or host_args.get("project", None)
    
    if project is not None and project != "":
        config_args = importlib.import_module(f"mon.createml.config.{project}.{config}")
    else:
        config_args = importlib.import_module(f"mon.createml.config.{config}")
    
    # Prioritize input args --> predefined args --> config file args
    project     = project or config_args.model["project"]
    project     = str(project).replace(".", "/")
    root        = root        or host_args.get("root",        None)
    name        = name        or host_args.get("name",        None)  or config_args.model["name"]
    weights     = weights     or host_args.get("weights",     None)  or config_args.model["weights"]
    batch_size  = batch_size  or host_args.get("batch_size",  None)  or config_args.data["batch_size"]
    image_size  = image_size  or host_args.get("image_size",  None)  or config_args.data["image_size"]
    accelerator = accelerator or host_args.get("accelerator", None)  or config_args.trainer["accelerator"]
    devices     = devices     or host_args.get("devices",     None)  or config_args.trainer["devices"]
    max_epochs  = max_epochs  or host_args.get("max_epochs",  None)  or config_args.trainer["max_epochs"]
    max_steps   = max_steps   or host_args.get("max_steps",   None)  or config_args.trainer["max_steps"]
    strategy    = strategy    or host_args.get("strategy",    None)  or config_args.trainer["strategy"]
    exist_ok    = exist_ok    or host_args.get("exist_ok",    False)
    
    # Update arguments
    args = mf.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["config_file"]  = config_args.__file__
    args["datamodule"]  |= {
        "image_size": image_size,
        "batch_size": batch_size,
    }
    args["model"]       |= {
        "weights" : weights,
        "fullname": name,
        "root"    : root,
        "project" : project,
    }
    args["trainer"]     |= {
        "accelerator": accelerator,
        "devices"    : devices,
        "max_epochs" : max_epochs,
        "max_steps"  : max_steps,
        "strategy"   : strategy,
    }
   
    if not exist_ok:
        mf.delete_dir(paths=mf.Path(root)/project/name)
        
    _train(args=args)


def _train(args: dict):
    # Initialization
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args['hostname']}")
    
    datamodule: nn.DataModule = DATAMODULES.build(config=args["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(phase="training")
    
    args["model"]["classlabels"] = datamodule.classlabels
    model: nn.Model          = MODELS.build(config=args["model"])
    model.phase                  = "training"

    mf.print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # Trainer
    console.rule("[bold red]2. SETUP TRAINER")
    mf.copy_file(src=args["config_file"], dst=model.root/"config.py")
    
    ckpt      = nn.get_latest_checkpoint(dirpath=model.ckpt_dir) if model.ckpt_dir.exists() else None
    callbacks = CALLBACKS.build_instances(configs=args["trainer"]["callbacks"])

    logger = []
    for k, v in args["trainer"]["logger"].items():
        if k == "tensorboard":
            v |= {"save_dir": model.root}
            logger.append(nn.TensorBoardLogger(**v))
    
    args["trainer"]["callbacks"]            = callbacks
    args["trainer"]["default_root_dir"]     = model.root
    args["trainer"]["enable_checkpointing"] = any(isinstance(cb, lcallbacks.Checkpoint) for cb in callbacks)
    args["trainer"]["logger"]               = logger
    args["trainer"]["num_sanity_val_steps"] = (0 if (ckpt is not None) else args["trainer"]["num_sanity_val_steps"])
    
    trainer               = nn.Trainer(**args["trainer"])
    trainer.current_epoch = nn.get_epoch(ckpt=ckpt)
    trainer.global_step   = nn.get_global_step(ckpt=ckpt)
    console.log("[green]Done")
    
    # Training
    console.rule("[bold red]3. TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = datamodule.train_dataloader,
        val_dataloaders   = datamodule.val_dataloader,
        ckpt_path         = ckpt,
    )
    console.log("[green]Done")
    
# endregion


# region Main

if __name__ == "__main__":
    train()

# endregion
