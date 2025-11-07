#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements QuadPrior model training pipeline for low-light image enhancement.

References:
    - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
      Priors," CVPR 2024.
    - Code: https://github.com/daooshee/QuadPrior
"""

import box
import pytorch_lightning as pl
import webdataset as wds
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy

import mon
from quadprior import (
    create_model, create_webdataset, disable_verbosity, ImageLogger,
    load_state_dict,
)

mon.init()
disable_verbosity()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.parse_device(args.device)
    device = mon.utils.to_int_list(device) if "auto" not in device else device

    # Seed
    mon.set_random_seed(args.seed)

    # Model
    cfg_path        = root_dir / "src" / "models" / args.cfg
    init_ckpt       = mon.ROOT_DIR / "zoo/cv/enhance/lle/quadprior/quadprior/coco/control_sd15_init.ckpt"
    pretrained_ckpt = mon.ROOT_DIR / "zoo/cv/enhance/lle/quadprior/quadprior/coco/control_sd15_coco_final.ckpt"
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model          = create_model(config_path=cfg_path).cpu()
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

    model.learning_rate    = args.optimizer.lr
    model.sd_locked        = args.network.sd_locked
    model.only_mid_control = args.network.only_mid_control
    
    # Callback
    logger = ImageLogger(save_dir=str(args.save_dir), batch_frequency=args.logger_freq)
    checkpoint_callback = ModelCheckpoint(
        dirpath                 = str(args.save_dir),
        filename                = args.fullname + "-{epoch:02d}-{step}",
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
        default_root_dir = str(args.save_dir),
        devices          = device,
        strategy         = "auto",  # strategy,
        # max_epochs       = epochs,
        max_steps        = args.epochs,
        precision        = 16,
        sync_batchnorm   = True,
        accelerator      = "gpu",
        callbacks        = [logger, checkpoint_callback],
    )
    
    # Data I/O
    data       = mon.data.parse_data_dir(args.root, data_dir=args.train_dataloader.dataset.root)
    dataset    = create_webdataset(data_dir=str(data))
    dataloader = wds.WebLoader(
        dataset         = dataset,
        batch_size      = args.batch_size,
        num_workers     = 2,
        pin_memory      = False,
        prefetch_factor = 2,
    )
    
    # Train
    trainer.fit(model, dataloader)

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
