#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the default configuration."""

from __future__ import annotations

__all__ = [
    "datamodule", "dataset", "debug", "learning_rate_monitor",
    "model_checkpoint", "rich_model_summary", "rich_progress_bar",
    "tensorboard", "trainer", "tune_report_callback", "tuner",
]

import albumentations as A

from mon.foundation import pathlib
from mon.globals import DATA_DIR

# region Callback

learning_rate_monitor = {
    "name"            : "learning_rate_monitor",
    "logging_interval": None,   # 'epoch' or 'step' (None = log at individual interval).
    "log_momentum"    : False,  # Log the momentum values of the optimizer.
}
"""
See Also: :class:`lightning.pytorch.callbacks.lr_monitor.LearningRateMonitor`
"""

model_checkpoint = {
    "name"                   : "model_checkpoint",
    "auto_insert_metric_name": True,   # The checkpoints filenames will contain the metric name.
    "dirpath"                : None,   # Directory to save the model file.
    "every_n_epochs"         : 1,      # Number of epochs between checkpoints.
    "every_n_train_steps"    : 0,      # Number of training steps between checkpoints (0 = disable).
    "filename"               : None,   # Checkpoint filename.
    "mode"                   : "min",  # 'min' or 'max'.
    "monitor"                : "checkpoint/loss/val_epoch",  # Quantity to monitor.
    "save_last"              : True,   # Save an exact copy of the checkpoint to a file `last.ckpt`.
    "save_on_train_epoch_end": None,   # Run checkpointing at the end of the training epoch.
    "save_top_k"             : 1,      # Save the best k models.
    "save_weights_only"      : True,   # Only the model's weights will be saved.
    "train_time_interval"    : None,   # Checkpoints are monitored at the specified time interval.
    "verbose"                : True,   # Verbosity mode.
}
"""
See Also: :class:`lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`
"""

rich_model_summary = {
    "name"     : "rich_model_summary",
    "max_depth": 2,  # The maximum depth of layer nesting that the summary will include (0 = disable).
}
"""
See Also: :class:`lightning.pytorch.callbacks.rich_model_summary.RichModelSummary`
"""

rich_progress_bar = {
    "name": "rich_progress_bar",
}
"""
See Also: :class:`lightning.pytorch.callbacks.progress.rich_progress.RichProgressBar`
"""

tune_report_callback = {
    "metrics": {"loss": "checkpoint/loss/val_epoch"},
    "on"     : "validation_end",
}
"""
See Also: :class:`ray.tune.integration.pytorch_lightning.TuneReportCallback`
"""

# endregion


# region Data

dataset = {
    "name"        : "a2i2-haze",       # Dataset/datamodule name.
    "split"       : None,              # 'train', 'test', 'predict' (None = load all).
    "root"        : DATA_DIR / "...",  # The root directory of the dataset.
    "image_size"  : 256,               # Image size in HW format (for resizing).
    "classlabels" : None,              # A file containing all class definitions.
    "transform"   : A.Compose(         # Transformations performing on both the input and target. 
        [
            A.Resize(width=256, height=256),
        ]
    ),
    "to_tensor"   : False,             # Convert input and target to torch.Tensor.
    "cache_data"  : False,             # Cache labels data to disk for faster loading next time.
    "cache_images": False,             # Cache images into memory for faster loading.
    "verbose"     : True,              # Verbosity.
}
"""
See Also: :class:`mon.coreml.data.dataset.Dataset`
"""

datamodule = {
    "name"        : "a2i2-haze",       # Dataset/datamodule name.
    "root"        : DATA_DIR / "...",  # The root directory of the dataset.
    "image_size"  : 256,               # Image size in HW format (for resizing).
    "transform"   : A.Compose(         # Transformations performing on both the input and target.
        [
            A.Resize(width=256, height=256),
        ]
    ),
    "to_tensor"   : False,             # Convert input and target to torch.Tensor.
    "cache_data"  : False,             # Cache labels data to disk for faster loading next time.
    "cache_images": False,             # Cache images into memory for faster loading.
    "batch_size"  : 8,                 # The number of samples in one forward pass.
    "devices"     : 0,                 # A list of devices to use.
    "shuffle"     : True,              # Reshuffle the datapoints at the beginning of every epoch.
    "verbose"     : True,              # Verbosity.
}
"""
See Also:
    :class:`mon.coreml.data.datamodule.DataModule` and
    :class:`mon.coreml.data.dataset.Dataset`
"""

# endregion


# region Logger

tensorboard = {
    "save_dir"         : pathlib.Path(),  # Save directory.
    "name"             : "",              # Experiment name ("" = no
    # per-experiment subdirectory).
    "version"          : "tensorboard",   # '/save_dir/name/version/'.
    "sub_dir"          : None,            # Subdirectory to group TensorBoard logs: '/save_dir/name/version/sub_dir/'
    "log_graph"        : False,           # Adds the computational graph to tensorboard.
    "default_hp_metric": True,
    "prefix"           : "",              # A string to put at the beginning of metric keys.
}
"""
See Also: :class:`lightning.pytorch.loggers.tensorboard.TensorBoardLogger`
"""

# endregion


# region Model

debug = {
    "every_best_epoch": True,   # Show only the best epochs.
    "every_n_epochs"  : 50,     # Number of epochs between debugging (0 = disable).
    "image_quality"   : 95,     # Image quality to be saved.
    "max_n"           : 8,      # Show max `n` images.
    "nrow"            : 8,      # The maximum number of items to display in a row
    "save"            : True,   # Save debug image
    "save_to_subdir"  : True,   # Save all debug images of the same epoch to a subdirectory naming after the epoch number.
    "wait_time"       : 0.001,  # Pause sometimes before showing the next image
}

# endregion


# region Training

trainer = {
    "accelerator"                      : "auto",  # 'cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'.
    "accumulate_grad_batches"          : None,    # Accumulates grads every k batches.
    "auto_lr_find"                     : False,   # Find the initial learning for faster convergence.
    "auto_scale_batch_size"            : False,   # Find the largest batch size that can fit into memory.
    "benchmark"                        : None,
    "callbacks"                        : None,    # Add a callback or list of callbacks.
    "check_val_every_n_epoch"          : 1,       # Run validation loop every after every `n` training epochs.
    "default_root_dir"                 : None,    # Default path for logs and weights.
    "detect_anomaly"                   : False,   # Enable anomaly detection for the autograd engine.
    "deterministic"                    : False,   # PyTorch operations must use deterministic algorithms.
    "devices"                          : None,    # Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`.
    "enable_checkpointing"             : True,
    "enable_model_summary"             : True,
    "enable_progress_bar"              : True,
    "fast_dev_run"                     : False,   # Run `n` batches of train, val, and test to find any bugs (True = 1).
    "gradient_clip_algorithm"          : "norm",  # 'value' or 'norm'.
    "gradient_clip_val"                : None,    # The value at which to clip gradients.
    "inference_mode"                   : False,
    "limit_predict_batches"            : 1.0,     # How much of prediction dataset to check (float = fraction, int = num_batches).
    "limit_test_batches"               : 1.0,     # How much of test dataset to check (float = fraction, int = num_batches).
    "limit_train_batches"              : 1.0,     # How much of training dataset to check (float = fraction, int = num_batches).
    "limit_val_batches"                : 1.0,     # How much of validation dataset to check (float = fraction, int = num_batches).
    "log_every_n_steps"                : 50,      # How often to log within steps.
    "logger"                           : True,    # Logger (or iterable collection of loggers) for experiment tracking (True = `TensorBoardLogger`).
    "max_epochs"                       : 500,     # -1: infinite training.
    "max_steps"                        : -1,      # -1: infinite training.
    "max_time"                         : None,    # Stop training after this amount of time has passed.
    "min_epochs"                       : None,    # Force training for at least these many epochs.
    "min_steps"                        : None,    # Force training for at least these many steps.
    "move_metrics_to_cpu"              : False,
    "multiple_trainloader_mode"        : "max_size_cycle",  # 'max_size_cycle' or 'min_size'.
    "num_nodes"                        : 1,       # Number of GPU nodes for distributed training.
    "num_sanity_val_steps"             : 0,       # Sanity check runs `n` validation batches before starting the training routine.
    "overfit_batches"                  : 0.0,     # Overfit training/validation data (float = fraction, int = num_batches).
    "plugins"                          : None,
    "precision"                        : 32,      # Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
    "profiler"                         : None,    # Profile individual steps during training and assist in identifying bottlenecks.
    "reload_dataloaders_every_n_epochs": 0,
    "replace_sampler_ddp"              : True,
    "strategy"                         : None,
    "sync_batchnorm"                   : False,   # Synchronize batch norm layers between process groups/whole world.
    "track_grad_norm"                  : -1,
    "val_check_interval"               : 1.0,     # How often to check the validation set (float = fraction of training epoch, int = fixed number of training batches).
}
"""
See Also: :class:`lightning.pytorch.trainer.trainer.Trainer`
"""

# endregion


# region Tuner

tuner = {
    "callbacks"            : None,   # List of callbacks.
    "chdir_to_trial_dir"   : True,   #
    "checkpoint_at_end"    : False,  # Whether to checkpoint at the end of the experiment.
    "checkpoint_freq"      : 0,      # How many training iterations between checkpoints (0 = disable).
    "checkpoint_score_attr": None,   # Specifies by which attribute to rank the best checkpoint.
    "config"               : {},
    "export_formats"       : None,   # List of formats that exported at the end of the experiment.
    "fail_fast"            : False,  # Whether to fail upon the first error.
    "keep_checkpoints_num" : None,   # Number of checkpoints to keep (None = keep all).
    "local_dir"            : "~/ray_results",  # Local dir to save training results.
    "log_to_file"          : False,
    "name"                 : None,   # Name of experiment.
    "num_samples"          : 1,      # Number of times to sample from the hyperparameter space.
    "max_concurrent_trials": None,   # Maximum number of trials to run concurrently.
    "max_failures"         : 0,      # Try to recover a trial at least this many times (-1 = infinite retries, 0 = disable).
    "metric"               : None,   # Metric to optimize.
    "mode"                 : "min",  # 'min' or 'max'.
    "progress_reporter"    : None,   #  Progress reporter for reporting intermediate experiment progress.
    "raise_on_failed_trial": True,
    "resources_per_trial"  : {},     # Machine resources to allocate per trial. Ex: {"cpu": 64, "gpu": 8}.
    "restore"              : None,   # Path to checkpoint.
    "resume"               : False,  # True, False, "LOCAL", "REMOTE", "PROMPT", "AUTO"
    "reuse_actors"         : None,   # Whether to reuse actors between different trials when possible.
    "run_or_experiment"    : None,
    "scheduler"            : None,   # Scheduler for executing the experiment. Refer to :mod:`ray.tune.schedulers` for more options.
    "search_alg"           : None,   # Search algorithm for optimization.
    "server_port"          : None,   # Port number for launching TuneServer.
    "stop"                 : None,   # Stopping criteria.
    "sync_config"          : None,   # Configuration object for syncing. See :class:`tune.SyncConfig`.
    "time_budget_s"        : None,   # Global time budget in seconds after which all trials are stopped.
    "trial_dirname_creator": None,
    "trial_executor"       : None,   # Manage the execution of trials.
    "trial_name_creator"   : None,
    "verbose"              : 3,      # 0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results.
}
"""
See Also: :mod:`ray.tune.tune.run`
"""

# endregion
