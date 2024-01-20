#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements training pipeline."""

from __future__ import annotations

import importlib
import random
import socket
from typing import Any

import click
from lightning.pytorch import callbacks as lcallbacks

import mon

console = mon.console


# region Function

def train(args: dict):
    # Initialization
    if mon.is_rank_zero():
        console.rule("[bold red]1. INITIALIZATION")
        console.log(f"Machine: {args['hostname']}")
    
    mon.set_random_seed(args["seed"])
    
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(phase="training")
    
    args["model"]["classlabels"] = datamodule.classlabels
    model: mon.Model             = mon.MODELS.build(config=args["model"])
    model.phase                  = "training"
    
    if mon.is_rank_zero():
        mon.print_dict(args, title=model.fullname)
        console.log("[green]Done")
    
    # Trainer
    if mon.is_rank_zero():
        console.rule("[bold red]2. SETUP TRAINER")
        mon.copy_file(src=args["config_file"], dst=model.root/"config.py")
    
    ckpt      = mon.get_latest_checkpoint(dirpath=model.ckpt_dir) if model.ckpt_dir.exists() else None
    callbacks = mon.CALLBACKS.build_instances(configs=args["trainer"]["callbacks"])

    logger = []
    for k, v in args["trainer"]["logger"].items():
        if k == "tensorboard":
            v |= {"save_dir": model.root}
            logger.append(mon.TensorBoardLogger(**v))
    
    args["trainer"]["callbacks"]            = callbacks
    args["trainer"]["default_root_dir"]     = model.root
    args["trainer"]["enable_checkpointing"] = any(isinstance(cb, lcallbacks.Checkpoint) for cb in callbacks)
    args["trainer"]["logger"]               = logger
    args["trainer"]["num_sanity_val_steps"] = (0 if (ckpt is not None) else args["trainer"]["num_sanity_val_steps"])
    
    trainer               = mon.Trainer(**args["trainer"])
    trainer.current_epoch = mon.get_epoch(ckpt=ckpt)
    trainer.global_step   = mon.get_global_step(ckpt=ckpt)
    if mon.is_rank_zero():
        console.log("[green]Done")
    
    # Training
    if mon.is_rank_zero():
        console.rule("[bold red]3. TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = datamodule.train_dataloader,
        val_dataloaders   = datamodule.val_dataloader,
        ckpt_path         = ckpt,
    )
    if mon.is_rank_zero():
        console.log(f"Model: {args['model']['fullname']}")  # Log
        console.log("[green]Done")
    

@click.command(context_settings=dict(
    ignore_unknown_options = True,
    allow_extra_args       = True,
))
@click.option("--config",      type=click.Path(exists=False), default=None,                help="The training config to use.")
@click.option("--name",        type=str,                      default=None,                help="Model name.")
@click.option("--variant",     type=str,                      default=None,                help="Model variant.")
@click.option("--data",        type=str,                      default=None,                help="Training dataset name.")
@click.option("--root",        type=click.Path(exists=True),  default=mon.RUN_DIR/"train", help="Save results to root/project/fullname.")
@click.option("--project",     type=click.Path(exists=False), default=None,                help="Save results to root/project/fullname.")
@click.option("--fullname",    type=str,                      default=None,                help="Save results to root/project/fullname.")
@click.option("--weights",     type=click.Path(exists=False), default=None,                help="Weights paths.")
@click.option("--batch-size",  type=int,                      default=None,                help="Total Batch size for all GPUs.")
@click.option("--image-size",  type=int,                      default=None,                help="Image sizes.")
@click.option("--seed",        type=int,                      default=100,                 help="Manual seed.")
@click.option("--accelerator", type=click.Choice(["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"], case_sensitive=False), default="gpu")
@click.option("--devices",                                    default="auto",              help="Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`.")
@click.option("--max-epochs",  type=int,                      default=100,                 help="Stop training once this number of epochs is reached.")
@click.option("--max-steps",   type=int,                      default=None,                help="Stop training once this number of steps is reached.")
@click.option("--strategy",    type=str,                      default="auto",              help="Supports different training strategies with aliases as well as custom strategies.")
@click.option("--exist-ok",    is_flag=True,                                               help="Whether to overwrite existing experiment.")
@click.option("--verbose",     is_flag=True)
@click.pass_context
def main(
    ctx,
    config     : str,
    name       : str,
    variant    : int | str | None,
    data       : str,
    root       : mon.Path,
    project    : str,
    fullname   : str | None,
    weights    : Any,
    batch_size : int,
    image_size : int | list[int],
    seed       : int,
    accelerator: str,
    devices    : int | str | list[int, str],
    max_epochs : int,
    max_steps  : int,
    strategy   : str,
    exist_ok   : bool,
    verbose    : bool,
):
    model_kwargs = {
        k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--"))
            else True for i, k in enumerate(ctx.args) if k.startswith("--")
    }
    
    # Get config module
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
    accelerator = accelerator or config_args.trainer["accelerator"]
    devices     = devices     or config_args.trainer["devices"]
    max_epochs  = max_epochs  or config_args.trainer["max_epochs"]
    max_steps   = max_steps   or config_args.trainer["max_steps"]
    strategy    = strategy    or config_args.trainer["strategy"]

    # Update arguments
    args                 = mon.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["root"]         = root
    args["project"]      = project
    args["fullname"]     = fullname
    args["image_size"]   = image_size
    args["seed"]         = seed
    args["verbose"]      = verbose
    args["config_file"]  = config_args.__file__
    args["datamodule"]  |= {
        "image_size": image_size,
        "batch_size": batch_size,
        "verbose"   : verbose,
    }
    args["model"] |= {
        "weights" : weights,
        "variant" : variant,
        "fullname": fullname,
        "root"    : root,
        "project" : project,
        "verbose" : verbose,
    }
    args["model"]   |= model_kwargs
    args["trainer"] |= {
        "accelerator": accelerator,
        "devices"    : devices,
        "max_epochs" : max_epochs,
        "max_steps"  : max_steps,
        "strategy"   : strategy,
    }
   
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(root) / project / name)
        
    train(args=args)

# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
