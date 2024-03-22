#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements some default configurations."""

from __future__ import annotations

__all__ = [
	"log_training_progress",
    "datamodule",
    "dataset",
    "learning_rate_monitor",
    "model_checkpoint",
    "predictor",
    "rich_model_summary",
    "rich_progress_bar",
    "tensorboard",
    "trainer",
    "tune_report_callback",
    "tuner",
]

# region Callback

learning_rate_monitor = {
    "name"            : "learning_rate_monitor",
    "logging_interval": None,   # 'epoch' or 'step' (None = log at individual interval).
    "log_momentum"    : False,  # Log the momentum values of the optimizer.
}

log_training_progress = {
	"name"                  : "log_training_progress",
	"dirpath"               : None,             # Directory to save the log file.
	"filename"              : "train_log.csv",  # Log filename.
	"every_n_epochs"        : 1,                # Log every n epochs.
	"every_n_train_steps"   : None,             # Log every n training steps.
	"train_time_interval"   : None,             # Log every n seconds.
	"log_on_train_epoch_end": None,             # Log on train epoch end.
	"verbose"               : True,             # Verbosity mode.
}

model_checkpoint = {
    "name"                   : "model_checkpoint",
	"dirpath"                : None,        # Directory to save the model file.
	"filename"               : None,        # Checkpoint filename.
	"monitor"                : "loss/val",  # Quantity to monitor.
	"mode"                   : "min",       # ``'min'`` or ``'max'``.
	"save_last"              : False,       # Save an exact copy of the checkpoint to a file `last.pt`.
	"save_top_k"             : 1,           # Save the best k models.
	"save_weights_only"      : False,       # Only the model's weights will be saved.
    "every_n_epochs"         : 1,           # Number of epochs between checkpoints.
    "every_n_train_steps"    : 0,           # Number of training steps between checkpoints (0 = disable).
	"train_time_interval"    : None,        # Checkpoints are monitored at the specified time interval.
    "save_on_train_epoch_end": True,        # Run checkpointing at the end of the training epoch.
	"auto_insert_metric_name": True,        # The checkpoints filenames will contain the metric name.
    "verbose"                : True,        # Verbosity mode.
}

rich_model_summary = {
    "name"     : "rich_model_summary",
    "max_depth": 2,  # The maximum depth of layer nesting that the summary will include (0 = disable).
}

rich_progress_bar = {
    "name"        : "rich_progress_bar",
    "refresh_rate": 1,
    "leave"       : False,
}

tune_report_callback = {
    "metrics": {"loss": "checkpoint/loss/val_epoch"},
    "on"     : "validation_end",
}

# endregion


# region Data

dataset = {
    "name"          : None,          # Dataset name.
    "split"         : None,          # ``'train'``, ``'test'``, ``'predict'`` (``None`` = load all).
    "root"          : None,          # The root directory of the dataset.
    "classlabels"   : None,          # A file containing all class definitions.
	"has_test_label": False,         # Whether the test dataset has ground-truth labels.
    "transform"     : None,          # Transformations performing on both the input and target.
    "to_tensor"     : False,         # Convert input and target to torch.Tensor.
    "cache_data"    : False,         # Cache labels data to disk for faster loading next time.
    "verbose"       : True,          # Verbosity.
}

datamodule = {
    "name"      : None,          # Datamodule name.
    "root"      : None,          # The root directory of the dataset.
    "transform" : None,          # Transformations performing on both the input and target.
    "to_tensor" : False,         # Convert input and target to torch.Tensor.
    "cache_data": False,         # Cache labels data to disk for faster loading next time.
    "batch_size": 8,             # The number of samples in one forward pass.
    "devices"   : 0,             # A list of devices to use.
    "shuffle"   : True,          # Reshuffle the datapoints at the beginning of every epoch.
    "verbose"   : True,          # Verbosity.
}

# endregion


# region Logger

tensorboard = {
    "save_dir"         : None,            # Save directory.
    "name"             : "",              # Experiment name ("" = no per-experiment subdirectory).
    "version"          : "tensorboard",   # '/save_dir/name/version/'.
    "sub_dir"          : None,            # Subdirectory to group TensorBoard logs: '/save_dir/name/version/sub_dir/'
    "log_graph"        : False,           # Adds the computational graph to tensorboard.
    "default_hp_metric": True,
    "prefix"           : "",              # A string to put at the beginning of metric keys.
}

# endregion


# region Training

trainer = {
    "accelerator"                      : "auto",  # 'cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'.
    "accumulate_grad_batches"          : 1,       # Accumulates grads every k batches.
    "benchmark"                        : None,
    "callbacks"                        : None,    # Add a callback or list of callbacks.
    "check_val_every_n_epoch"          : 1,       # Run validation loop every after every `n` training epochs.
    "default_root_dir"                 : None,    # Default path for logs and weights.
    "detect_anomaly"                   : False,   # Enable anomaly detection for the autograd engine.
    "deterministic"                    : False,   # PyTorch operations must use deterministic algorithms.
    "devices"                          : "auto",  # Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`.
    "enable_checkpointing"             : True,
    "enable_model_summary"             : True,
    "enable_progress_bar"              : True,
    "fast_dev_run"                     : False,   # Run `n` batches of train, val, and test to find any bugs (True = 1).
    "gradient_clip_algorithm"          : "norm",  # 'value' or 'norm'.
    "gradient_clip_val"                : "0.1",   # The value at which to clip gradients.
    "inference_mode"                   : False,
    "limit_predict_batches"            : 1.0,     # How much of prediction dataset to check (float = fraction, int = num_batches).
    "limit_test_batches"               : 1.0,     # How much of test dataset to check (float = fraction, int = num_batches).
    "limit_train_batches"              : 1.0,     # How much of training dataset to check (float = fraction, int = num_batches).
    "limit_val_batches"                : 1.0,     # How much of validation dataset to check (float = fraction, int = num_batches).
    "log_every_n_steps"                : 1,       # How often to log within steps.
    "logger"                           : True,    # Logger (or iterable collection of loggers) for experiment tracking (True = `TensorBoardLogger`).
    "max_epochs"                       : 500,     # -1: infinite training.
    "max_steps"                        : -1,      # -1: infinite training.
    "max_time"                         : None,    # Stop training after this amount of time has passed.
    "min_epochs"                       : None,    # Force training for at least these many epochs.
    "min_steps"                        : None,    # Force training for at least these many steps.
    "num_nodes"                        : 1,       # Number of GPU nodes for distributed training.
    "num_sanity_val_steps"             : 0,       # Sanity check runs `n` validation batches before starting the training routine.
    "overfit_batches"                  : 0.0,     # Overfit training/validation data (float = fraction, int = num_batches).
    "plugins"                          : None,
    "precision"                        : 32,      # Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
    "profiler"                         : None,    # Profile individual steps during training and assist in identifying bottlenecks.
    "reload_dataloaders_every_n_epochs": 0,
    "strategy"                         : "auto",  # 'ddp', 'ddp_spawn', 'ddp2', 'ddp_cpu', 'ddp2_cpu', 'ddp_spawn_cpu', 'horovod', 'tpu', 'tpu_spawn', 'auto'.
    "sync_batchnorm"                   : False,   # Synchronize batch norm layers between process groups/whole world.
    "use_distributed_sampler"          : True,
    "val_check_interval"               : 1.0,     # How often to check the validation set (float = fraction of training epoch, int = fixed number of training batches).
}

# endregion


# region Predicting

predictor = {
	"source"		  : None,    # Source data.
	"default_root_dir": None,    # Default path for saving results.
	"devices"         : "auto",  # Running devices.
	"augment"         : None ,   # Test-time augmentation.
    "benchmark"       : False,   # Measure efficient score.
    "save_image"      : False,   # Save result images.
    "verbose"         : True,    # Verbosity.
}

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
    "mode"                 : "min",  # ``'min'`` or ``'max'``.
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

# endregion
