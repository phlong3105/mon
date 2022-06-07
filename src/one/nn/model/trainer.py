#!/usr/bin/env python
# -*- coding: utf-8 -*-qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in

"""Add-in to the `pytorch_lightning.Trainer` class.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.accelerators import HPUAccelerator
from pytorch_lightning.accelerators import IPUAccelerator
from pytorch_lightning.accelerators import TPUAccelerator
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities import _IPU_AVAILABLE
from pytorch_lightning.utilities import _TPU_AVAILABLE

from one.core import console

__all__ = [
    "Trainer"
]


# MARK: - Trainer

class Trainer(pl.Trainer):
    """Override `pytorch_lightning.Trainer` with several methods and properties.
    """
    
    # MARK: Properties
    
    @pl.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        self.fit_loop.current_epoch = current_epoch
    
    @pl.Trainer.global_step.setter
    def global_step(self, global_step: int):
        self.fit_loop.global_step = global_step
        
    # MARK: Configure

    def _log_device_info(self):
        console.log(f"GPU available: {torch.cuda.is_available()}, "
                    f"used: {isinstance(self.accelerator, GPUAccelerator)}")
    
        num_tpu_cores = self.num_devices if isinstance(self.accelerator, TPUAccelerator) else 0
        console.log(f"TPU available: {_TPU_AVAILABLE}, using: {num_tpu_cores} TPU cores")
    
        num_ipus = self.num_devices if isinstance(self.accelerator, IPUAccelerator) else 0
        console.log(f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs")
    
        num_hpus = self.num_devices if isinstance(self.accelerator, HPUAccelerator) else 0
        console.log(f"HPU available: {_HPU_AVAILABLE}, using: {num_hpus} HPUs")
    
        if torch.cuda.is_available() and not isinstance(self.accelerator, GPUAccelerator):
            console.log(
                "GPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='gpu', devices={GPUAccelerator.auto_device_count()})`.",
            )
    
        if _TPU_AVAILABLE and not isinstance(self.accelerator, TPUAccelerator):
            console.log(
                "TPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='tpu', devices={TPUAccelerator.auto_device_count()})`."
            )
    
        if _IPU_AVAILABLE and not isinstance(self.accelerator, IPUAccelerator):
            console.log(
                "IPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='ipu', devices={IPUAccelerator.auto_device_count()})`."
            )
    
        if _HPU_AVAILABLE and not isinstance(self.accelerator, HPUAccelerator):
            console.log(
                "HPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='hpu', devices={HPUAccelerator.auto_device_count()})`."
            )
