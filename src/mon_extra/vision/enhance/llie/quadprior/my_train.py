#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from cldm.hack import disable_verbosity

disable_verbosity()
import argparse

import pytorch_lightning as pl
import webdataset as wds
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy

import mon
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from coco_dataset import create_webdataset

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(args: argparse.Namespace):
    # General config
    fullname         = args.fullname
    save_dir         = mon.Path(args.save_dir)
    weights          = args.weights
    device           = mon.parse_device(args.device)
    device           = mon.to_int_list(device) if "auto" not in device else device
    epochs           = args.epochs
    steps            = args.steps
    
    batch_size       = args.batch_size
    num_workers      = args.num_workers
    prefetch_factor  = args.prefetch_factor
    learning_rate    = float(args.lr)
    logger_freq      = args.logger_freq
    sd_locked        = args.sd_locked
    only_mid_control = args.only_mid_control
    verbose          = args.verbose
    
    config_path      = _current_dir / args.config_path  # "./models/cldm_v15.yaml"
    init_ckpt        = mon.ZOO_DIR / "vision/enhance/llie/quadprior/quadprior/coco/control_sd15_init.ckpt"
    pretrained_ckpt  = mon.ZOO_DIR / "vision/enhance/llie/quadprior/quadprior/coco/control_sd15_coco_final.ckpt"
    
    # Directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model          = create_model(config_path=config_path).cpu()
    state_dict     = load_state_dict(str(init_ckpt), location="cpu")
    new_state_dict = {}
    for s in state_dict:
        if "cond_stage_model.transformer" not in s:
            new_state_dict[s] = state_dict[s]
    model.load_state_dict(new_state_dict)
    model.add_new_layers()
    
    if pretrained_ckpt != "":
        state_dict = load_state_dict(str(pretrained_ckpt), location="cpu")
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if "_forward_module.control_model" in sd_name:
            new_state_dict[sd_name.replace("_forward_module.control_model.", "")] = sd_param
    model.control_model.load_state_dict(new_state_dict)
    
    model.learning_rate    = learning_rate
    model.sd_locked        = sd_locked
    model.only_mid_control = only_mid_control
    
    # Data I/O
    data       = mon.DATA_DIR / args.data_dir
    dataset    = create_webdataset(data_dir=str(data))
    dataloader = wds.WebLoader(
        dataset         = dataset,
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = False,
        prefetch_factor = prefetch_factor,
    )
    
    # Callback
    logger = ImageLogger(save_dir=str(save_dir), batch_frequency=logger_freq)
    checkpoint_callback = ModelCheckpoint(
        dirpath                 = str(save_dir),
        filename                = fullname + "-{epoch:02d}-{step}",
        # filename                = fullname,
        monitor                 = "step",
        save_last               = False,
        save_top_k              = -1,
        verbose                 = True,
        every_n_train_steps     = 10000,  # How frequent to save checkpoint
        save_on_train_epoch_end = True,
    )
    
    # Trainer
    strategy = DeepSpeedStrategy(
        stage             = 2,
        offload_optimizer = True,
        cpu_checkpointing = True
    )
    trainer = pl.Trainer(
        default_root_dir = str(save_dir),
        devices          = device,
        strategy         = "auto",  # strategy,
        # max_epochs       = epochs,
        max_steps        = steps,
        precision        = 16,
        sync_batchnorm   = True,
        accelerator      = "gpu",
        callbacks        = [logger, checkpoint_callback],
    )
    
    # Train
    trainer.fit(model, dataloader)

# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=_current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
