#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the training procedure."""

from __future__ import annotations

__all__ = [
    "Trainer", "seed_everything",
]

import lightning
from lightning.pytorch.trainer import *
from lightning.pytorch.utilities import _HPU_AVAILABLE, _IPU_AVAILABLE

from mon.coreml import strategy
from mon.foundation import console


# region Trainer

class Trainer(lightning.Trainer):
    """The trainer class that extend the :class:`lightning.Trainer` with several
    methods and properties.
    
    See Also: :class:`lightning.Trainer`.
    """
    
    @lightning.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        self.fit_loop.current_epoch = current_epoch
    
    @lightning.Trainer.global_step.setter
    def global_step(self, global_step: int):
        self.fit_loop.global_step = global_step
    
    def _log_device_info(self):
        if strategy.CUDAAccelerator.is_available():
            gpu_available = True
            gpu_type      = " (cuda)"
        elif strategy.MPSAccelerator.is_available():
            gpu_available = True
            gpu_type      = " (mps)"
        else:
            gpu_available = False
            gpu_type      = ""
        
        gpu_used = isinstance(self.accelerator, (strategy.CUDAAccelerator, strategy.MPSAccelerator))
        console.log(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}.")
        
        num_tpu_cores = self.num_devices if isinstance(self.accelerator, strategy.TPUAccelerator) else 0
        console.log(f"TPU available: {strategy.TPUAccelerator.is_available()}, using: {num_tpu_cores} TPU cores.")

        num_ipus = self.num_devices if isinstance(self.accelerator, strategy.IPUAccelerator) else 0
        console.log(f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs.")
        
        num_hpus = self.num_devices if isinstance(self.accelerator, strategy.HPUAccelerator) else 0
        console.log(f"HPU available: {_HPU_AVAILABLE}, using: {num_hpus} HPUs.")
        
        # Integrate MPS Accelerator here, once gpu maps to both
        if strategy.CUDAAccelerator.is_available() and not isinstance(self.accelerator, strategy.CUDAAccelerator):
            console.log(
                f"GPU available but not used. Set 'accelerator' and 'devices' "
                f"using 'Trainer(accelerator='gpu', devices="
                f"{strategy.CUDAAccelerator.auto_device_count()})'.",
            )
        
        if strategy.TPUAccelerator.is_available() and not isinstance(self.accelerator, strategy.TPUAccelerator):
            console.log(
                f"TPU available but not used. Set `accelerator` and `devices` "
                f"using `Trainer(accelerator='tpu', devices="
                f"{strategy.TPUAccelerator.auto_device_count()})`."
            )
        
        if _IPU_AVAILABLE and not isinstance(self.accelerator, strategy.IPUAccelerator):
            console.log(
                f"IPU available but not used. Set 'accelerator' and 'devices' "
                f"using 'Trainer(accelerator='ipu', devices="
                f"{strategy.IPUAccelerator.auto_device_count()})'."
            )
        
        if _HPU_AVAILABLE and not isinstance(self.accelerator, strategy.HPUAccelerator):
            console.log(
                f"HPU available but not used. Set 'accelerator' and 'devices' "
                f"using 'Trainer(accelerator='hpu', devices="
                f"{strategy.HPUAccelerator.auto_device_count()})'."
            )
        
        if strategy.MPSAccelerator.is_available() and not isinstance(self.accelerator, strategy.MPSAccelerator):
            console.log(
                f"MPS available but not used. Set 'accelerator' and 'devices' "
                f"using 'Trainer(accelerator='mps', devices="
                f"{strategy.MPSAccelerator.auto_device_count()})'."
            )

# endregion
