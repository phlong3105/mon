#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements testing pipeline."""

from __future__ import annotations

import importlib
import socket
from typing import Any

import click
from lightning.pytorch import callbacks as lcallbacks

from mon import coreml, foundation as mf
from mon.globals import CALLBACKS, DATAMODULES, MODELS, RUN_DIR

console = mf.console


# region Host

hosts = {
	"lp-labdesktop01-ubuntu": {
		"config"     : "",
        "root"       : RUN_DIR/"train",
        "project"    : None,
        "name"       : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (256, 256),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
    "vsw-ws02": {
		"config"     : "",
        "root"       : RUN_DIR/"train",
        "project"    : None,
        "name"       : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (256, 256),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
    "vsw-ws03": {
		"config"     : "",
        "root"       : RUN_DIR/"train",
        "project"    : None,
        "name"       : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (256, 256),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
}

# endregion


# region Function

@click.command()
@click.option("--config",      default="",    type=click.Path(exists=False), help="The training config to use.")
@click.option("--root",        default=RUN_DIR/"train", type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--project",     default=None,  type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--name",        default=None,  type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--weights",     default=None,  type=click.Path(exists=False), help="Weights paths.")
@click.option("--batch-size",  default=8,     type=int, help="Total Batch size for all GPUs.")
@click.option("--image-size",  default=None,  type=int, nargs="+", help="Image sizes.")
@click.option("--accelerator", default="gpu", type=click.Choice(["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"], case_sensitive=False))
@click.option("--devices",     default=0,     type=int, help="Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`.")
@click.option("--max-epochs",  default=None,  type=int, help="Stop training once this number of epochs is reached.")
@click.option("--max-steps",   default=None,  type=int, help="Stop training once this number of steps is reached.")
@click.option("--strategy",    default=None,  type=int, help="Supports different training strategies with aliases as well custom strategies.")
def test(
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
    name        = name        or host_args.get("name",        None) or config_args.model["name"]
    weights     = weights     or host_args.get("weights",     None) or config_args.model["weights"]
    batch_size  = batch_size  or host_args.get("batch_size",  None) or config_args.data["batch_size"]
    image_size  = image_size  or host_args.get("image_size",  None) or config_args.data["image_size"]
    accelerator = accelerator or host_args.get("accelerator", None) or config_args.trainer["accelerator"]
    devices     = devices     or host_args.get("devices",     None) or config_args.trainer["devices"]
    max_epochs  = max_epochs  or host_args.get("max_epochs",  None) or config_args.trainer["max_epochs"]
    max_steps   = max_steps   or host_args.get("max_steps",   None) or config_args.trainer["max_steps"]
    strategy    = strategy    or host_args.get("strategy",    None) or config_args.trainer["strategy"]
    
    # Update arguments
    args = mf.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["config_file"]  = config_args.__file__,
    args["data"]        |= {
        "image_size": image_size,
        "batch_size": batch_size,
    }
    args["model"]       |= {
        "weights": weights,
        "name"   : name,
        "root"   : root,
        "project": project,
    }
    args["trainer"]     |= {
        "accelerator": accelerator,
        "devices"    : devices,
        "max_epochs" : max_epochs,
        "max_steps"  : max_steps,
        "strategy"   : strategy,
    }
    _test(args=args)


def _test(args: dict):
    # Initialization
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args['hostname']}")
    datamodule: coreml.DataModule = DATAMODULES.build(config=args["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(phase="testing")
    
    args["model"]["classlabels"] = datamodule.classlabels
    model: coreml.Model          = MODELS.build(config=args["model"])
    
    mf.print_dict(args, title=model.fullname)
    console.log("[green]Done")
    
    # Trainer
    console.rule("[bold red]2. SETUP TRAINER")
    mf.copy_file(src=args["config_file"], dst=model.root)
    
    ckpt      = coreml.get_latest_checkpoint(dirpath=model.weights_dir)
    callbacks = CALLBACKS.build_instances(configs=args["trainer"]["callbacks"])
    enable_checkpointing = any(isinstance(cb, lcallbacks.Checkpoint) for cb in callbacks)
    
    logger = []
    for k, v in args["trainer"]["logger"].items():
        if k == "tensorboard":
            v |= {"save_dir": model.root}
            logger.append(coreml.TensorBoardLogger(**v))
    
    args["trainer"]["callbacks"]            = callbacks
    args["trainer"]["default_root_dir"]     = model.root
    args["trainer"]["enable_checkpointing"] = enable_checkpointing
    args["trainer"]["logger"]               = logger
    
    trainer = coreml.Trainer(**args["trainer"])
    console.log("[green]Done")
    
    # Training
    console.rule("[bold red]3. TESTING")
    results = trainer.test(
        model       = model,
        dataloaders = datamodule.test_dataloader,
    )
    console.log(results)
    console.log("[green]Done")
    
# endregion


# region Main

if __name__ == "__main__":
    test()

# endregion
