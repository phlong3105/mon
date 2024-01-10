#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements training pipeline."""

from __future__ import annotations

import importlib
import socket
from typing import Any

import click
from lightning.pytorch import callbacks as lcallbacks

import mon

console = mon.console


# region Host

hosts = {
	"lp-labdesktop-01": {
        "config"     : "zid_jin2022",
        "root"       : mon.RUN_DIR / "train",
        "project"    : "vision/enhance/dehaze/zid",
        "name"       : "zid-jin2022",
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 1,
        "image_size" : (512, 512),
        "accelerator": "auto",
        "devices"    : "auto",
        "max_epochs" : 500,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
        "verbose"    : True,
	},
    "vsw-ws01" : {
        "config"     : "transweather_gtrain",
        "root"       : mon.RUN_DIR / "train",
        "project"    : "vision/enhance/derain/transweather",
        "name"       : "transweather-gtrain",
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 32,
        "image_size" : (512, 512),
        "accelerator": "auto",
        "devices"    : [0, 1],
        "max_epochs" : None,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
        "verbose"    : True,
	},
    "vsw-ws02" : {
        "config"     : "finet_i_haze",
        "root"       : mon.RUN_DIR / "train",
        "project"    : "vision/enhance/universal/finet",
        "name"       : "finet-i-haze",
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 4,
        "image_size" : (256, 256),
        "accelerator": "auto",
        "devices"    : "auto",
        "max_epochs" : 500,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
        "verbose"    : True,
	},
    "vsw-ws-03": {
        "config"     : "zeroadce_sice_zerodce",
        "root"       : mon.RUN_DIR / "train",
        "project"    : "vision/enhance/llie/zeroadce",
        "name"       : "zeroadce-sice-zerodce",
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 32,
        "image_size" : (512, 512),
        "accelerator": "auto",
        "devices"    : "auto",
        "max_epochs" : 300,
        "max_steps"  : None,
        "strategy"   : "auto",
        "exist_ok"   : False,
        "verbose"    : True,
	},
}

# endregion


# region Function

def train(args: dict):
    # Initialization
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args['hostname']}")
    
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(phase="training")
    
    args["model"]["classlabels"] = datamodule.classlabels
    model: mon.Model             = mon.MODELS.build(config=args["model"])
    model.phase                  = "training"

    mon.print_dict(args, title=model.fullname)
    console.log("[green]Done")

    # Trainer
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
    console.log("[green]Done")
    
    # Training
    console.rule("[bold red]3. TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = datamodule.train_dataloader,
        val_dataloaders   = datamodule.val_dataloader,
        ckpt_path         = ckpt,
    )
    console.log(f"Model: {args['model']['fullname']}")  # Log
    console.log("[green]Done")
    

@click.command(context_settings=dict(
    ignore_unknown_options = True,
    allow_extra_args       = True,
))
@click.option("--config",      type=click.Path(exists=False),  default="",                 help="The training config to use.")
@click.option("--root",        type=click.Path(exists=True),   default=mon.RUN_DIR/"train",help="Save results to root/project/name.")
@click.option("--project",     type=click.Path(exists=False),  default=None,               help="Save results to root/project/name.")
@click.option("--name",        type=click.Path(exists=False),  default=None,               help="Save results to root/project/name.")
@click.option("--variant",     type=str,                       default=None,               help="Variant.")
@click.option("--weights",     type=click.Path(exists=False),  default=None,               help="Weights paths.")
@click.option("--batch-size",  type=int,                       default=None,               help="Total Batch size for all GPUs.")
@click.option("--image-size",  type=int,                       default=None,               help="Image sizes.")
@click.option("--accelerator", type=click.Choice(["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"], case_sensitive=False), default="gpu")
@click.option("--devices",     type=int,                       default=0,                  help="Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`.")
@click.option("--max-epochs",  type=int,                       default=100,                help="Stop training once this number of epochs is reached.")
@click.option("--max-steps",   type=int,                       default=None,               help="Stop training once this number of steps is reached.")
@click.option("--strategy",    type=str,                       default="auto",             help="Supports different training strategies with aliases as well as custom strategies.")
@click.option("--exist-ok",    is_flag=True,                                               help="Whether to overwrite existing experiment.")
@click.option("--verbose",     is_flag=True)
@click.pass_context
def main(
    ctx,
    config     : mon.Path | str,
    root       : mon.Path,
    project    : str,
    name       : str,
    variant    : int | str | None,
    weights    : Any,
    batch_size : int,
    image_size : int | list[int],
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
    
    # Obtain arguments
    hostname  = socket.gethostname().lower()
    host_args = hosts[hostname]
    config    = config  or host_args.get("config",  None)
    project   = project or host_args.get("project", None)
    
    if project is not None and project != "":
        project_module = project.replace("/", ".")
        config_args    = importlib.import_module(f"config.{project_module}.{config}")
    else:
        config_args    = importlib.import_module(f"config.{config}")
    
    # Prioritize input args --> predefined args --> config file args
    project     = project or config_args.model["project"]
    project     = str(project).replace(".", "/")
    root        = root        or host_args.get("root",        None)
    name        = name        or host_args.get("name",        None)  or config_args.model["name"]
    variant     = variant     or host_args.get("variant",     None)  or config_args.model["variant"]
    variant     = None if variant in ["", "none", "None"] else variant
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
    args                 = mon.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["root"]         = root
    args["project"]      = project
    args["image_size"]   = image_size
    args["verbose"]      = verbose
    args["config_file"]  = config_args.__file__
    args["datamodule"]  |= {
        "image_size": image_size,
        "batch_size": batch_size,
    }
    args["model"] |= {
        "weights" : weights,
        "variant" : variant,
        "fullname": name,
        "root"    : root,
        "project" : project,
    }
    args["model"]   |= model_kwargs
    args["trainer"] |= {
        "accelerator": accelerator,
        "devices"    : devices,
        "max_epochs" : max_epochs,
        "max_steps"  : max_steps,
        "strategy"   : mon.DDPStrategy(find_unused_parameters=True),
    }
   
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(root) / project / name)
        
    train(args=args)

# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
