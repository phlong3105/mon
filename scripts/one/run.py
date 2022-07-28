#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import socket
from copy import deepcopy

from munch import Munch
from pytorch_lightning.loggers import TensorBoardLogger

from one import BaseModel
from one import CALLBACKS
from one import CheckpointCallback
from one import console
from one import copy_config_file
from one import DataModule
from one import DATAMODULES
from one import EvalOutput
from one import get_epoch
from one import get_global_step
from one import get_latest_checkpoint
from one import Inference
from one import load_config
from one import MODELS
from one import ModelState
from one import print_dict
from one import set_distributed_backend
from one import Trainer
from scripts.host import hosts


# MARK: - Main

def main():
    """Main function."""
    # NOTE: Initialization
    console.rule("[bold red]1. INITIALIZATION")
    hostname = socket.gethostname().lower()
    host     = hosts[hostname]
    console.log(f"Host: {hostname}")
    
    # Distributed backend
    set_distributed_backend(strategy=host.strategy)
    
    # Configs
    config                          = load_config(config=host.config.config)
    config.data.gpus                = host.gpus
    config.trainer.amp_backend      = host.amp_backend
    config.trainer.strategy         = host.strategy
    config.trainer.gpus             = host.gpus
    config.trainer.auto_select_gpus = (True if isinstance(host.gpus, int) else False)
    
    # Data
    dm = DATAMODULES.build_from_dict(cfg=config.data)
    dm.prepare_data()

    # Model
    config.model.classes = dm.classes
    model                     = MODELS.build_from_dict(cfg=config.model)
    
    print_dict(config, title=host.config.model_fullname)
    console.log("[green]Done")
    
    # NOTE: Training
    if host.model_state is ModelState.TRAINING:
        console.rule("[bold red]2. TRAINING DATA PREPARATION")
        copy_config_file(host.config.__file__, model.version_dir)
        dm.setup()
        console.log("[green]Done")
        
        console.rule("[bold red]3. MODEL TRAINING")
        model.model_state = host.model_state
        train(model=model, dm=dm, config=config)
        console.log("[green]Done")
        
    # NOTE: Testing
    if host.model_state in [ModelState.TRAINING, ModelState.TESTING]:
        console.rule("[bold red]2. TESTING DATA PREPARATION")
        dm.setup(phase=host.model_state)
        console.log("[green]Done")
        
        console.rule("[bold red]3. MODEL TESTING")
        model.model_state = host.model_state
        test(model=model, dm=dm, config=config)
        console.log("[green]Done")
        
    # NOTE: Inference
    if host.model_state is ModelState.INFERENCE:
        console.rule("[bold red]3. MODEL INFERENCE")
        model.model_state = host.model_state
        infer(model=model, data=host.infer_data, config=config)
        console.log("[green]Done")
        

# MARK: - Training

def train(model: BaseModel, dm: DataModule, config: Munch) -> BaseModel:
    """Train the model.
    
    Args:
        model (BaseModel):
            Model.
        dm (DataModule):
            Datamodule.
        config (Munch):
            Config dictionary.

    Returns:
        model (BaseModel):
            Trained model.
    """
    _cfg = deepcopy(config)
    
    # NOTE: Get weights
    ckpt_name = f"*last*.ckpt"
    ckpt      = get_latest_checkpoint(dirpath=model.weights_dir, name=ckpt_name)
    
    # NOTE: Callbacks
    callbacks            = CALLBACKS.build_from_dictlist(cfgs=_cfg.callbacks)
    enable_checkpointing = any(isinstance(cb, CheckpointCallback) for cb in callbacks)
    
    # NOTE: Logger
    tb_logger = TensorBoardLogger(**_cfg.tb_logger)
    
    # NOTE: Trainer
    trainer_cfg                      = _cfg.trainer
    trainer_cfg.default_root_dir     = model.version_dir
    trainer_cfg.callbacks            = callbacks
    trainer_cfg.enable_checkpointing = enable_checkpointing
    trainer_cfg.logger               = tb_logger
    trainer_cfg.num_sanity_val_steps = (0 if (ckpt is not None) else trainer_cfg.num_sanity_val_steps)
    
    trainer               = Trainer(**trainer_cfg)
    trainer.current_epoch = get_epoch(ckpt=ckpt)
    trainer.global_step   = get_global_step(ckpt=ckpt)
    
    # NOTE: Train
    trainer.fit(
        model             = model,
        train_dataloaders = dm.train_dataloader,
        val_dataloaders   = dm.val_dataloader,
        ckpt_path         = ckpt,
    )
    return model


# MARK: - Testing

def test(model: BaseModel, dm: DataModule, config: Munch) -> EvalOutput:
    """Test the model.
    
    Args:
        model (BaseModel):
            Model.
        dm (DataModule):
            Datamodule.
        config (Munch):
            Config dictionary.
    
    Returns:
        results (EvalOutput):
            List of dictionaries with metrics logged during the test model_state,
            e.g., in model- or callback hooks like :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.test_epoch_end`,
            etc. The length of the list corresponds to the number of test
            dataloaders used.
    """
    _cfg = deepcopy(config)
    
    # NOTE: Get weights
    ckpt_name = f"*best*.ckpt"
    ckpt      = get_latest_checkpoint(dirpath=model.weights_dir, name=ckpt_name)

    # NOTE: Callbacks
    callbacks = CALLBACKS.build_from_dictlist(cfgs=_cfg.callbacks)

    # NOTE: Logger
    tb_logger = TensorBoardLogger(**_cfg.tb_logger)
    
    # NOTE: Trainer
    trainer_cfg                  = _cfg.trainer
    trainer_cfg.default_root_dir = model.version_dir
    trainer_cfg.callbacks        = callbacks
    trainer_cfg.logger           = tb_logger
    
    trainer               = Trainer(**trainer_cfg)
    trainer.current_epoch = get_epoch(ckpt=ckpt)
    trainer.global_step   = get_global_step(ckpt=ckpt)
    
    # NOTE: Test
    results = trainer.test_realblurj(
        model       = model,
        dataloaders = dm.test_dataloader,
        ckpt_path   = ckpt
    )
    print(results)
    return results


# MARK: - Inference

def infer(model: BaseModel, data: str, config: Munch):
    """Inference.
    
    Args:
        model (BaseModel):
            Model.
        data (str):
            Data source.
        config (Munch):
            Config dictionary.
    """
    _cfg = deepcopy(config)
    
    # NOTE: Get weights
    ckpt_name = f"*best*.ckpt"
    ckpt      = get_latest_checkpoint(dirpath=model.weights_dir, name=ckpt_name)
    if ckpt:
        model = model.load_from_checkpoint(checkpoint_path=ckpt, **_cfg.model)
    
    # NOTE: Inference
    inference_cfg                  = _cfg.inference
    inference_cfg.default_root_dir = os.path.join(model.version_dir, "infer")
    
    inference = Inference(**inference_cfg)
    
    # NOTE: Infer
    model.model_state = ModelState.INFERENCE
    inference.run(model=model, data=data)


if __name__ == "__main__":
    main()
