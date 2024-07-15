#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements training pipeline."""

from __future__ import annotations

import socket

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

def parse_train_args(model_root: str | mon.Path | None = None) -> dict:
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(mon.parse_train_input_args())
    config     = input_args.get("config")
    root       = mon.Path(input_args.get("root"))
    
    # Get config args
    config = mon.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = None,
        config       = config,
    )
    args   = mon.load_config(config)
    
    # Prioritize input args --> config file args
    arch       = input_args.get("arch")
    model      = input_args.get("model")    or args.get("model_name")
    data       = input_args.get("data")     or args.get("data_name")
    root       = root                       or args.get("root")
    project    = input_args.get("project")  or args.get("project")
    variant    = input_args.get("variant")  or args.get("variant")
    fullname   = input_args.get("fullname") or args.get("fullname")
    save_dir   = input_args.get("save_dir") or args.get("save_dir")
    weights    = input_args.get("weights")  or args["model"]["weights"]
    devices    = input_args.get("device")   or args["trainer"]["devices"]
    local_rank = input_args.get("local_rank")
    launcher   = input_args.get("launcher")
    epochs     = input_args.get("epochs")
    epochs     = epochs     if epochs > 0 else args["trainer"]["max_epochs"]
    steps      = input_args.get("steps")
    steps      = steps      if steps  > 0 else args["trainer"]["max_steps"]
    exist_ok   = input_args.get("exist_ok") or args.get("exist_ok")
    verbose    = input_args.get("verbose")  or args.get("verbose")
    extra_args = input_args.get("extra_args")
    
    # Parse arguments
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    weights  = None       if isinstance(weights, list | tuple) and len(weights) == 0 else weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
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
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config is not None and config.is_config_file():
        mon.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        
    return args


def main():
    args = parse_train_args()
    train(args)


if __name__ == "__main__":
    main()

# endregion
