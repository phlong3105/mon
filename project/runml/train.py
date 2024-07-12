#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements training pipeline."""

from __future__ import annotations

import socket

import click
from lightning.pytorch import callbacks as lcallbacks

import mon
import mon.core.utils

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(args: dict) -> str:
    # General config
    if mon.is_rank_zero():
        console.rule("[bold red]1. INITIALIZATION")
        console.log(f"Machine: {args['hostname']}")
    
    # Seed
    mon.set_random_seed(args["seed"])
    
    # Model
    model: mon.Model = mon.MODELS.build(config=args["model"])
    if mon.is_rank_zero():
        mon.print_dict(args, title=model.fullname)
        console.log("[green]Done")
    
    # Data I/O
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(phase="training")
    
    # Trainer
    if mon.is_rank_zero():
        console.rule("[bold red]2. SETUP TRAINER")
    mon.copy_file(src=args["config"], dst=model.root / "config.py")
    
    ckpt      = mon.get_latest_checkpoint(dirpath=model.ckpt_dir) if model.ckpt_dir.exists() else None
    callbacks = mon.CALLBACKS.build_instances(configs=args["trainer"]["callbacks"])

    logger = []
    for k, v in args["trainer"]["logger"].items():
        if k == "tensorboard":
            v |= {"save_dir": model.root}
            logger.append(mon.TensorBoardLogger(**v))
    
    save_dir = args["trainer"]["default_root_dir"]
    save_dir = save_dir if save_dir not in [None, "None", ""] else model.root
    args["trainer"]["callbacks"]            = callbacks
    args["trainer"]["default_root_dir"]     = save_dir
    args["trainer"]["enable_checkpointing"] = any(isinstance(cb, lcallbacks.Checkpoint) for cb in callbacks)
    args["trainer"]["logger"]               = logger
    args["trainer"]["num_sanity_val_steps"] = (0 if (ckpt is not None) else args["trainer"]["num_sanity_val_steps"])
    
    trainer               = mon.Trainer(**args["trainer"])
    trainer.current_epoch = mon.get_epoch_from_checkpoint(ckpt=ckpt)
    trainer.global_step   = mon.get_global_step_from_checkpoint(ckpt=ckpt)
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
    
    return str(save_dir)
    
# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config",   type=str, default=None, help="Model config.")
@click.option("--arch",     type=str, default=None, help="Model architecture.")
@click.option("--model",    type=str, default=None, help="Model name.")
@click.option("--root",     type=str, default=None, help="Project root.")
@click.option("--project",  type=str, default=None, help="Project name.")
@click.option("--variant",  type=str, default=None, help="Variant name.")
@click.option("--fullname", type=str, default=None, help="Fullname to save the model's weight.")
@click.option("--save-dir", type=str, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
@click.option("--weights",  type=str, default=None, help="Weights paths.")
@click.option("--device",   type=str, default=None, help="Running devices.")
@click.option("--epochs",   type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",    type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok", is_flag=True)
@click.option("--verbose",  is_flag=True)
def main(
    config  : str,
    arch    : str,
    model   : str,
    root    : str,
    project : str,
    variant : str,
    fullname: str,
    save_dir: str,
    weights : str,
    device  : str,
    epochs  : int,
    steps   : int,
    exist_ok: bool,
    verbose : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = mon.load_config(config)
    
    # Prioritize input args --> config file args
    model    = model    or args["model_name"]
    data     =             args["data_name"]
    root     = root     or args["root"]
    project  = project  or args["project"]
    variant  = variant  or args["variant"]
    fullname = fullname or args["fullname"]
    weights  = weights  or args["model"]["weights"]
    devices  = device   or args["trainer"]["devices"]
    epochs   = epochs   if epochs > 0 else args["trainer"]["max_epochs"]
    steps    = steps    if steps  > 0 else args["trainer"]["max_steps"]
    
    # Parse arguments
    root     = mon.Path(root)
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    weights  = weights[0] if isinstance(weights, list | tuple) else weights
    devices  = mon.parse_device(devices)
    devices  = mon.to_int_list(devices) if "auto" not in devices else devices
    
    # Update arguments
    args["hostname"]  = hostname
    args["config"]    = config
    args["arch"]      = arch
    args["root"]      = root
    args["project"]   = project
    args["variant"]   = variant
    args["fullname"]  = fullname
    args["verbose"]   = verbose
    args["model"]    |= {
        "root"    : save_dir,
        "fullname": fullname,
        "weights" : weights,
        "verbose" : verbose,
    }
    args["trainer"]  |= {
        "default_root_dir": save_dir,
        "devices"         : devices,
        "max_epochs"      : epochs if steps is not None else None,
        "max_steps"       : steps,
    }
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(save_dir))
   
    return train(args=args)


if __name__ == "__main__":
    main()

# endregion
