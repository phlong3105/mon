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
    
    Args:
        accelerator: Supports passing different accelerator types ("cpu", "gpu",
            "tpu", "ipu", "hpu", "mps, "auto") as well as custom accelerator
            instances.
        accumulate_grad_batches: Accumulates grads every k batches or as set up
            in the dict. Defaults to None.
        amp_backend: The mixed precision backend to use ("native" or "apex").
            Defaults to "native".
        amp_level: The optimization level to use (O1, O2, etc...). By default
            it will be set to "O2" if `amp_backend='apex'`.
        auto_lr_find: If set to True, will make :meth:`trainer.tune()` run a
            learning rate finder, trying to optimize initial learning for faster
            convergence. :meth:`trainer.tune` method will set the suggested
            learning rate in :attr:`self.lr` or :attr:`self.learning_rate` in
            the :class:`lightning.LightningModule`. To use a different key set a
            string instead of True with the key name. Defaults to False.
        auto_scale_batch_size: If set to True, will initially run a batch size
            finder trying to find the largest batch size that fits into memory.
            The result will be stored in :atr:`self.batch_size` in the
            :class:`lightning.LightningModule`. Additionally, can be set to
            either `power` that estimates the batch size through a power search
            or `binsearch` that estimates the batch size through a binary
            search. Defaults to False.
        auto_select_gpus: If enabled and `gpus` or `devices` is an integer,
            pick available gpus automatically. This is especially useful when
            GPUs are configured to be in "exclusive mode", such that only one
            process at a time can access them. Defaults to False.
        benchmark: The value (True or False) to set
            `torch.backends.cudnn.benchmark` to. The value for
            `torch.backends.cudnn.benchmark` set in the current session will be
            used (False if not manually set). If `deterministic` is set to True,
            this will default to False. Override to manually set a different
            value. Defaults to None.
        callbacks: Add a callback or list of callbacks. Defaults to None.
        check_val_every_n_epoch: Perform a validation loop every after every n
            training epochs. If None, validation will be done solely based on
            the number of training batches, requiring :atr:`val_check_interval`
            to be an integer value. Defaults to 1.
        default_root_dir: Default path for logs and weights when no
            logger/ckpt_callback passed. Can be remote file paths such as
            `s3://mybucket/path` or 'hdfs://path/'. Defaults to os.getcwd().
        detect_anomaly: Enable anomaly detection for the autograd engine.
            Defaults to False.
        deterministic: If True, sets whether PyTorch operations must use
            deterministic algorithms. Set to "warn" to use deterministic
            algorithms whenever possible, throwing warnings on operations that
            don't support deterministic mode (requires PyTorch 1.11+). If not
            set, defaults to False. Defaults to None.
        devices: Will be mapped to either gpus, tpu_cores, num_processes or
            ipus, based on the accelerator type.
        enable_checkpointing: If True, enable checkpointing. It will configure a
            default
            :class:`mon.coreml.callback.model_checkpoint.ModelCheckpoint`
            callback if there is no user-defined
            :class:`mon.coreml.callback.model_checkpoint.ModelCheckpoint` in
            param:`callbacks`. Defaults to True.
        enable_model_summary: Whether to enable model summarization by default.
            Defaults to True.
        enable_progress_bar: Whether to enable to progress bar by default.
            Defaults to True.
        fast_dev_run: Runs n if set to `n` (int) else 1 if set to True batch(es)
            of train, val and test to find any bugs (ie: a sort of unit test).
            Defaults to False.
        gradient_clip_val: The value at which to clip gradients. Passing
            `gradient_clip_val=None` disables gradient clipping. If using
            Automatic Mixed Precision (AMP), the gradients will be unscaled
            before. Defaults to None.
        gradient_clip_algorithm: The gradient clipping algorithm to use.
            Pass `gradient_clip_algorithm="value"` to clip by value, and
            `gradient_clip_algorithm="norm"` to clip by norm. By default,
            it will be set to "norm".
        limit_train_batches: How much of training dataset to check
            (float = fraction, int = num_batches). Defaults to 1.0.
        limit_val_batches: How much of validation dataset to check
            (float = fraction, int = num_batches). Defaults to 1.0.
        limit_test_batches: How much of test dataset to check
            (float = fraction, int = num_batches). Defaults to 1.0.
        limit_predict_batches: How much of prediction dataset to check
            (float = fraction, int = num_batches). Defaults to 1.0.
        logger: Logger (or iterable collection of loggers) for experiment
            tracking.
            - If True uses the default `TensorBoardLogger`.
            - If False will disable logging.
            - If multiple loggers are provided and the `save_dir` property of
              that logger is not set, local files (checkpoints, profiler traces,
              etc.) are saved in `default_root_dir` rather than in the `log_dir`
              of the individual loggers.
            Defaults to True.
        log_every_n_steps: How often to log within steps. Defaults to 50.
        max_epochs: Stop training once this number of epochs is reached.
            Disabled by default (None).
            - If both `max_epochs` and `max_steps` are not specified, defaults
              to `max_epochs=1000`.
            - To enable infinite training, set `max_epochs=-1`.
        min_epochs: Force training for at least these many epochs. Disabled by
            default (None).
        max_steps: Stop training after this number of steps. Disabled by
            default (-1).
            - If `max_steps= 1` and `max_epochs=None`, will default  to
              `max_epochs = 1000`.
            - To enable infinite training, set `max_epochs=-1`.
        min_steps: Force training for at least these number of steps.
            Disabled by default (None).
        max_time: Stop training after this amount of time has passed. Disabled
            by default (None). The time duration can be specified in the format
            DD:HH:MM:SS (days, hours, minutes seconds), as a
            :class:`datetime.timedelta`, or a dictionary with keys that will be
            passed to :class:`datetime.timedelta`.
        move_metrics_to_cpu: Whether to force internal logged metrics to be
            moved to cpu. This can save some gpu memory, but can make training
            slower. Use with attention. Defaults to False.
        multiple_trainloader_mode: How to loop over the datasets when there are
            multiple train loaders.
            - In `max_size_cycle` mode, the trainer ends one epoch when the
              largest dataset is traversed, and smaller datasets reload when
              running out of their data.
            - In `min_size` mode, all the datasets reload when reaching the
              minimum length of datasets.
            Defaults to "max_size_cycle".
        num_nodes: Number of GPU nodes for distributed training. Defaults to 1.
        num_sanity_val_steps: Sanity check runs n validation batches before
            starting the training routine. Set it to -1 to run all batches in
            all validation dataloaders. Defaults to 2.
        overfit_batches: Over-fit a fraction of training/validation data (float)
            or a set number of batches (int). Defaults to 0.0.
        plugins: Plugins allow modification of core behavior like ddp and amp,
            and enable custom lightning plugins. Defaults to None.
        precision: Double precision (64), full precision (32), half precision
            (16) or bfloat16 precision (bf16). Can be used on CPU, GPU, TPUs,
            HPUs or IPUs. Defaults to 32.
        profiler: To profile individual steps during training and assist in
            identifying bottlenecks. Defaults to None.
        reload_dataloaders_every_n_epochs: Set to a non-negative integer to
            reload dataloaders every n epochs. Defaults to 0.
        replace_sampler_ddp: Explicitly enables or disables sampler replacement.
            If not specified this will toggle automatically when DDP is used. By
            default, it will add `shuffle=True` for train sampler and
            `shuffle=False` for val/test sampler. If you want to customize it,
            you can set `replace_sampler_ddp=False` and add your own distributed
            sampler.
        strategy: Supports different training strategies with aliases as well
            custom strategies. Defaults to None.
        sync_batchnorm: Synchronize batch norm layers between process
            groups/whole world. Defaults to False.
        track_grad_norm:
            - -1 no tracking. Otherwise tracks that p-norm.
            - May be set to 'inf' infinity-norm.
            - If using Automatic Mixed Precision (AMP), the gradients will be
              unscaled before logging them.
            Defaults to -1.
        val_check_interval: How often to check the validation set.
            - Pass a `float` in the range [0.0, 1.0] to check after a fraction
              of the training epoch.
            - Pass an `int` to check after a fixed number of training batches.
              An `int` value can only be higher than the number of training
              batches when `check_val_every_n_epoch=None`, which validates
              after every `N` training batches across epochs or during
              iteration-based training.
            Defaults to 1.0.
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
            gpu_type = " (cuda)"
        elif strategy.MPSAccelerator.is_available():
            gpu_available = True
            gpu_type = " (mps)"
        else:
            gpu_available = False
            gpu_type = ""
        
        gpu_used = isinstance(
            self.accelerator, (strategy.CUDAAccelerator, strategy.MPSAccelerator)
        )
        console.log(
            f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}."
        )
        
        num_ipus = self.num_devices if isinstance(
            self.accelerator, strategy.IPUAccelerator
        ) else 0
        console.log(
            f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs."
        )
        
        num_hpus = self.num_devices if isinstance(
            self.accelerator, strategy.HPUAccelerator
        ) else 0
        console.log(
            f"HPU available: {_HPU_AVAILABLE}, using: {num_hpus} HPUs."
        )
        
        # Integrate MPS Accelerator here, once gpu maps to both
        if strategy.CUDAAccelerator.is_available() and not isinstance(
            self.accelerator, strategy.CUDAAccelerator
        ):
            console.log(
                f"GPU available but not used. Set `accelerator` and `devices` "
                f"using `Trainer(accelerator='gpu', devices="
                f"{strategy.CUDAAccelerator.auto_device_count()})`.",
            )
        
        if _IPU_AVAILABLE and not isinstance(self.accelerator, strategy.IPUAccelerator):
            console.log(
                f"IPU available but not used. Set `accelerator` and `devices` "
                f"using `Trainer(accelerator='ipu', devices="
                f"{strategy.IPUAccelerator.auto_device_count()})`."
            )
        
        if _HPU_AVAILABLE and not isinstance(self.accelerator, strategy.HPUAccelerator):
            console.log(
                f"HPU available but not used. Set `accelerator` and `devices` "
                f"using `Trainer(accelerator='hpu', devices="
                f"{strategy.HPUAccelerator.auto_device_count()})`."
            )
        
        if strategy.MPSAccelerator.is_available() and not isinstance(
            self.accelerator, strategy.MPSAccelerator
        ):
            console.log(
                f"MPS available but not used. Set `accelerator` and `devices` "
                f"using `Trainer(accelerator='mps', devices="
                f"{strategy.MPSAccelerator.auto_device_count()})`."
            )

# endregion
