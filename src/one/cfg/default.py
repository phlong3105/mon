#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Default configs for each component.
"""

from __future__ import annotations

from one.constants import RUNS_DIR


# H1: - Callback ---------------------------------------------------------------

learning_rate_monitor = {
	"name": "learning_rate_monitor",
		# Name of the callback.
	"logging_interval": None,
		# Set to `epoch` or `step` to log lr of all optimizers at the same
		# interval, set to None to log at individual interval according to the
		# interval key of each scheduler. Defaults to None.
	"log_momentum": False,
		# Option to also log the momentum values of the optimizer, if the
		# optimizer has the momentum or betas attribute. Defaults to False.
}


model_checkpoint = {
	"name": "model_checkpoint",
	    # Name of the callback.
    "root": RUNS_DIR,
        # Root directory to save checkpoint files
	"filename": None,
        # Checkpoint filename. Can contain named formatting options to be
        # autofilled. Defaults to None and will be set to {epoch}-{step}.
    "monitor": "checkpoint/loss/val_epoch",  # "loss_epoch",
	    # Quantity to monitor. Defaults to None which saves a checkpoint only
		# for the last epoch.
	"save_last": True,
        # When True, saves an exact copy of the checkpoint to a file `last.ckpt`
		# whenever a checkpoint file gets saved. This allows accessing the
		# latest checkpoint in a deterministic manner. Defaults to None.
	"save_top_k": 5,
        # - If `save_top_k == k`, the best k models according to the quantity
		#   monitored will be saved.
        # - If `save_top_k == 0`, no models are saved.
        # - If `save_top_k == -1`, all models are saved. Please note that the
		#   monitors are checked every `every_n_epochs` epochs.
        # - If `save_top_k >= 2` and the callback is called multiple times
        #   inside an epoch, the name of the saved file will be appended with
		#   a version count starting with `v1`.
	"save_on_train_epoch_end": True,
        # Whether to run checkpointing at the end of the training epoch.
        # If this is False, then the check runs at the end of the validation.
		# Defaults to False.
	"save_weights_only": False,
		# If True, then only the model's weights will be saved. Otherwise,
	    # the optimizer states, lr-scheduler states, etc. are added in the
	    # checkpoint too.
	"mode": "min",
		# One of {min, max}. If `save_top_k != 0`, the decision to overwrite
		# the current save file is made based on either the maximization or the
		# minimization of the monitored quantity. For `val_acc`, this should be
		# `max`, for `val_loss` this should be `min`, etc.
	"every_n_train_steps": None,
	    # Number of training steps between checkpoints.
	    # - If `every_n_train_steps == None` or `every_n_train_steps == 0`,
	    #   we skip saving during training.
	    # - To disable, set `every_n_train_steps = 0`. This value must be None
		#   or non-negative.
	    # - This must be mutually exclusive with `train_time_interval` and
	    #   `every_n_epochs`.
	"train_time_interval": None,
		# Checkpoints are monitored at the specified time interval. For all
	    # practical purposes, this cannot be smaller than the amount of
	    # time it takes to process a single training batch. This is not
	    # guaranteed to execute at the exact time specified, but should be
	    # close. This must be mutually exclusive with `every_n_train_steps`
	    # and `every_n_epochs`.
	"every_n_epochs": 1,
		# Number of epochs between checkpoints. This value must be None or
	    # non-negative.
	    # - To disable saving top-k checkpoints, set `every_n_epochs = 0`.
	    # - This argument does not impact the saving of `save_last=True`
	    #   checkpoints.
	    # - If all of `every_n_epochs`, `every_n_train_steps` and
	    #   `train_time_interval` are None, we save a checkpoint at the end
	    #   of every epoch (equivalent to `every_n_epochs = 1`).
	    # - If `every_n_epochs == None` and either
	    #   `every_n_train_steps != None` or `train_time_interval != None`,
	    #   saving at the end of each epoch is disabled (equivalent to
	    #   `every_n_epochs = 0`).
	    # - This must be mutually exclusive with `every_n_train_steps` and
	    #   `train_time_interval`.
	    # - Setting both `ModelCheckpoint(..., every_n_epochs=V, save_on_train_epoch_end=False)`
	    #   and `Trainer(max_epochs=N, check_val_every_n_epoch=M)` will only
	    #   save checkpoints at epochs 0 < E <= N where both values for
	    #   `every_n_epochs` and `check_val_every_n_epoch` evenly divide E.
	"auto_insert_metric_name": True,
		# - When True, the checkpoints filenames will contain the metric name.
		#   For example, `filename='checkpoint_{epoch:02d}-{acc:02.0f}` with
		#   epoch `1` and acc `1.12` will resolve to
		#   `checkpoint_epoch=01-acc=01.ckpt`.
	    # - Is useful to set it to False when metric names contain `/` as this
		#   will result in extra folders. For example,
	    #   `filename='epoch={epoch}-step={step}-val_acc={val/acc:.2f}', auto_insert_metric_name=False`
	"verbose": True,
		#   Verbosity mode. Defaults to False.
}


rich_model_summary = {
	"name": "rich_model_summary",
		# Name of the callback.
	"max_depth": 2,
		# Maximum depth of layer nesting that the summary will include. A value
		# of 0 turns the layer summary off.
}


rich_progress_bar = {
	"name": "rich_progress_bar",
		# Name of the callback.
}


# H1: - Logger -----------------------------------------------------------------

tensorboard = {
	"save_dir": RUNS_DIR,
	    # Save directory.
	"name": "",
	    # Experiment name. Default: `default`. If it is the empty string then
		# no per-experiment subdirectory is used.
	"version": "tensorboard",
	    # Experiment version. If version is not specified the logger inspects
		# the save directory for existing versions, then automatically assigns
		# the next available version. If it is a string then it is used as the
		# run-specific subdirectory name, otherwise `version_${version}` is used.
	"sub_dir": None,
	    # Subdirectory to group TensorBoard logs. If a sub_dir argument is
		# passed then logs are saved in `/save_dir/version/sub_dir/`.
		# Defaults to None in which logs are saved in `/save_dir/version/`.
	"log_graph": False,
	    # Adds the computational graph to tensorboard. This requires that the
		# user has defined the `self.example_input_array` attribute in their
		# model.
	"default_hp_metric": True,
	    # Enables a placeholder metric with key `hp_metric` when
		# `log_hyperparams` is called without a metric (otherwise calls to
		# log_hyperparams without a metric are ignored).
	"prefix": "",
	    # A string to put at the beginning of metric keys.
}


# H1: - Trainer ----------------------------------------------------------------

debug = {
	"every_n_epochs": 10,
        # Number of epochs between debugging. To disable, set `every_n_epochs=0`.
		# Defaults to 1.
    "save_to_subdir": True,
        # Save all debug images of the same epoch to a subdirectory naming
	    # after the epoch number. Defaults to True.
    "image_quality": 95,
        # Image quality to be saved. Defaults to 95.
    "max_n": 8,
        # Show max n images if `input` has a batch size of more than `max_n`
		# items. Defaults to None means show all.
    "nrow": 8,
        # The maximum number of items to display in a row. The final grid size
		# is (n / nrow, nrow). If None, then the number of items in a row will
		# be the same as the number of items in the list. Defaults to 8.
    "wait_time": 0.001,
        # Pause sometimes before showing the next image. Defaults to 0.001.
	"save": True,
		# Save debug image. Defaults to False.
}


trainer = {
	"accelerator": "auto",
		# Supports passing different accelerator types ("cpu", "gpu", "tpu",
		# "ipu", "hpu", "mps", "auto") as well as custom accelerator instances.
	"accumulate_grad_batches": None,
        # Accumulates grads every k batches or as set up in the dict.
        # Defaults to None.
	"amp_backend": "native",
	    # The mixed precision backend to use ("native" or "apex").
        # Defaults to "native".
	"amp_level": None,
        # The optimization level to use (O1, O2, etc...). By default it will be
        # set to "O2" if `amp_backend='apex'`.
	"auto_lr_find": False,
        # If set to True, will make `trainer.tune()` run a learning rate finder,
        # trying to optimize initial learning for faster convergence.
        # `trainer.tune()` method will set the suggested learning rate in
        # `self.lr` or `self.learning_rate` in the `LightningModule`. To use a
        # different key set a string instead of True with the key name.
        # Defaults to False.
	"auto_scale_batch_size": False,
        # If set to True, will `initially` run a batch size finder trying to
        # find the largest batch size that fits into memory. The result will
        # be stored in `self.batch_size` in the `LightningModule`. Additionally,
        # can be set to either `power` that estimates the batch size through a
        # power search or `binsearch` that estimates the batch size through a
        # binary search. Defaults to False.
	"auto_select_gpus": False,
        # If enabled and `gpus` or `devices` is an integer, pick available gpus
        # automatically. This is especially useful when GPUs are configured to
        # be in "exclusive mode", such that only one process at a time can
        # access them. Defaults to False.
	"benchmark": None,
        # The value (True or False) to set `torch.backends.cudnn.benchmark` to.
        # The value for `torch.backends.cudnn.benchmark` set in the current
        # session will be used (False if not manually set). If `deterministic`
        # is set to True, this will default to False. Override to manually set
        # a different value. Defaults to None.
	"callbacks": None,
	    # Add a callback or list of callbacks. Defaults to None.
	"check_val_every_n_epoch": 1,
	    # Perform a validation loop every after every n training epochs.
        # If None, validation will be done solely based on the number of
        # training batches, requiring `val_check_interval` to be an integer
        # value. Defaults to 1.
	"default_root_dir": None,
        # Default path for logs and weights when no logger/ckpt_callback passed.
        # Can be remote file paths such as `s3://mybucket/path` or
        # 'hdfs://path/'. Defaults to os.getcwd().
	"detect_anomaly": False,
        # Enable anomaly detection for the autograd engine. Defaults to False.
	"deterministic": False,
        # If True, sets whether PyTorch operations must use deterministic
        # algorithms. Set to "warn" to use deterministic algorithms whenever
        # possible, throwing warnings on operations that don't support
        # deterministic mode (requires PyTorch 1.11+). If not set, defaults to
        # False. Defaults to None.
	"devices": None,
	    # Will be mapped to either gpus, tpu_cores, num_processes or ipus,
        # based on the accelerator type.
	"enable_checkpointing": True,
        # If True, enable checkpointing. It will configure a default
        # `ModelCheckpoint` callback if there is no user-defined
        # `ModelCheckpoint` in `callbacks`. Defaults to True.
	"enable_model_summary": True,
	    # Whether to enable model summarization by default. Defaults to True.
    "enable_progress_bar": True,
        # Whether to enable to progress bar by default. Defaults to True.
	"fast_dev_run": False,
        # Runs n if set to `n` (int) else 1 if set to True batch(es) of train,
        # val and test to find any bugs (ie: a sort of unit test).
        # Defaults to False.
	"gradient_clip_val": 0.1,
        # The value at which to clip gradients. Passing `gradient_clip_val=None`
        # disables gradient clipping. If using Automatic Mixed Precision (AMP),
        # the gradients will be unscaled before. Defaults to None.
	"gradient_clip_algorithm": "norm",
        # The gradient clipping algorithm to use.
        # Pass `gradient_clip_algorithm="value"` to clip by  value, and
        # `gradient_clip_algorithm="norm"` to clip by norm. By default, it will
        # be set to "norm".
	"limit_train_batches": 1.0,
	    # How much of training dataset to check (float = fraction,
        # int = num_batches). Defaults to 1.0.
	"limit_val_batches": 1.0,
        # How much of validation dataset to check (float = fraction,
        # int = num_batches). Defaults to 1.0.
	"limit_test_batches": 1.0,
	    # How much of test dataset to check (float = fraction,
        # int = num_batches). Defaults to 1.0.
	"limit_predict_batches": 1.0,
        # How much of prediction dataset to check (float = fraction,
        # int = num_batches). Defaults to 1.0.
	"logger": True,
        # Logger (or iterable collection of loggers) for experiment tracking.
        # - If True uses the default `TensorBoardLogger`.
        # - If False will disable logging.
        # - If multiple loggers are provided and the `save_dir` property of
        #   that logger is not set, local files (checkpoints, profiler traces,
        #   etc.) are saved in `default_root_dir` rather than in the `log_dir`
        #   of the individual loggers.
        # Defaults to True.
	"log_every_n_steps": 50,
	    # How often to log within steps. Defaults to 50.
	"max_epochs": 500,
        # Stop training once this number of epochs is reached.
        # Disabled by default (None).
        # - If both `max_epochs` and `max_steps` are not specified, defaults
        #   to `max_epochs=1000`.
        # - To enable infinite training, set `max_epochs=-1`.
    "min_epochs": None,
        # Force training for at least these many epochs.
        # Disabled by default (None).
	"max_steps": -1,
	    # Stop training after this number of steps. Disabled by default (-1).
        # - If `max_steps= 1` and `max_epochs=None`, will default  to
        #   `max_epochs = 1000`.
        # - To enable infinite training, set `max_epochs=-1`.
    "min_steps": None,
	    # Force training for at least these number of steps.
        # Disabled by default (None).
	"max_time": None,
        # Stop training after this amount of time has passed. Disabled by
        # default (None). The time duration can be specified in the format
        # DD:HH:MM:SS (days, hours, minutes seconds), as a
        # :class:`datetime.timedelta`, or a dictionary with keys that will be
        # passed to :class:`datetime.timedelta`.
	"move_metrics_to_cpu": False,
        # Whether to force internal logged metrics to be moved to cpu. This can
        # save some gpu memory, but can make training slower. Use with attention.
        # Defaults to False.
	"multiple_trainloader_mode": "max_size_cycle",
        # How to loop over the datasets when there are multiple train loaders.
        # - In `max_size_cycle` mode, the trainer ends one epoch when the
        #   largest dataset is traversed, and smaller datasets reload when
        #   running out of their data.
        # - In `min_size` mode, all the datasets reload when reaching the
        #   minimum length of datasets.
        # Defaults to "max_size_cycle".
	"num_nodes": 1,
        # Number of GPU nodes for distributed training. Defaults to 1.
	"num_sanity_val_steps": 0,
        # Sanity check runs n validation batches before starting the training
        # routine. Set it to -1 to run all batches in all validation
        # dataloaders. Defaults to 2.
	"overfit_batches": 0.0,
	    # Over-fit a fraction of training/validation data (float) or a set
        # number of batches (int). Defaults to 0.0.
    "plugins": None,
        # Plugins allow modification of core behavior like ddp and amp, and
        # enable custom lightning plugins. Defaults to None.
    "precision": 32,
        # Double precision (64), full precision (32), half precision (16) or
        # bfloat16 precision (bf16). Can be used on CPU, GPU, TPUs, HPUs or
        # IPUs. Defaults to 32.
	"profiler": None,
        # To profile individual steps during training and assist in identifying
        # bottlenecks. Defaults to None.
	"reload_dataloaders_every_n_epochs": 0,
        # Set to a non-negative integer to reload dataloaders every n epochs.
        # Defaults to 0.
	"replace_sampler_ddp": True,
        # Explicitly enables or disables sampler replacement. If not specified
        # this will toggle automatically when DDP is used. By default, it will
        # add `shuffle=True` for train sampler and `shuffle=False` for val/test
        # sampler. If you want to customize it, you can set
        # `replace_sampler_ddp=False` and add your own distributed sampler.
	"strategy": None,
        # Supports different training strategies with aliases as well custom
        # strategies. Defaults to None.
	"sync_batchnorm": False,
        # Synchronize batch norm layers between process groups/whole world.
        # Defaults to False.
	"track_grad_norm": -1,
	    # - -1 no tracking. Otherwise tracks that p-norm.
        # - May be set to 'inf' infinity-norm.
        # - If using Automatic Mixed Precision (AMP), the gradients will be
        #   unscaled before logging them.
        # Defaults to -1.
	"val_check_interval": 1.0,
        # How often to check the validation set.
        # - Pass a `float` in the range [0.0, 1.0] to check after a fraction
        #   of the training epoch.
        # - Pass an `int` to check after a fixed number of training batches.
        #   An `int` value can only be higher than the number of training
        #   batches when `check_val_every_n_epoch=None`, which validates after
        #   every `N` training batches across epochs or during iteration-based
        #   training.
        # Defaults to 1.0.
}
