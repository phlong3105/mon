#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zero-DCE trained on LIME dataset.
"""

from __future__ import annotations

from one.cfg import default
from one.constants import RUNS_DIR
from one.constants import VISION_BACKEND
from one.vision.transformation import Resize


# H1: - Basic ------------------------------------------------------------------

model_name = "finet-dehaze"
model_cfg  = "finet"
data_name  = "nhhaze"
fullname   = f"{model_name}-{data_name}"
root       = RUNS_DIR / "train"
project    = "finet"
shape      = [3, 256, 256]


# H1: - Data -------------------------------------------------------------------

data = {
    "name": data_name,
        # Dataset's name.
    "shape": shape,
        # Image shape as [C, H, W], [H, W], or [S, S].
    "transform": None,
        # Functions/transforms that takes in an input sample and returns a
        # transformed version.
    "target_transform": None,
        # Functions/transforms that takes in a target and returns a transformed
        # version.
    "transforms": [
        Resize(size=shape),
    ],
        # Functions/transforms that takes in an input and a target and returns
        # the transformed versions of both.
    "cache_data": False,
        # If True, cache data to disk for faster loading next time.
        # Defaults to False.
    "cache_images": False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Defaults to False.
    "backend": VISION_BACKEND,
        # Vision backend to process image. Defaults to VISION_BACKEND.
    "batch_size": 4,
        # Number of samples in one forward & backward pass. Defaults to 1.
    "devices" : 0,
        # The devices to use. Defaults to 0.
    "shuffle": True,
        # If True, reshuffle the data at every training epoch. Defaults to True.
    "verbose": True,
        # Verbosity. Defaults to True.
}


# H1: - Model ------------------------------------------------------------------

model = {
	"cfg": model_cfg,
        # Model's layers configuration. It can be an external .yaml path or a
        # dictionary. Defaults to None means you should define each layer
        # manually in `self.parse_model()` method.
    "root": root,
        # The root directory of the model. Defaults to RUNS_DIR.
    "project": project,
		# Project name. Defaults to None.
    "name": model_name,
        # Model's name. In case None is given, it will be
        # `self.__class__.__name__`. Defaults to None.
    "fullname": fullname,
        # Model's fullname in the following format: {name}-{data_name}-{postfix}.
        # In case None is given, it will be `self.basename`. Defaults to None.
    "channels": 3,
        # Input channel. Defaults to 3.
    "num_classes": None,
        # Number of classes for classification or detection tasks.
        # Defaults to None.
    "classlabels": None,
        # ClassLabels object that contains all labels in the dataset.
        # Defaults to None.
    "phase": "training",
        # Model's running phase. Defaults to training.
    "pretrained": None,
        # Initialize weights from pretrained.
    "loss": {"name": "psnr_loss", "max_val": 1.0, "weight": 0.5},
        # Loss function for training model. Defaults to None.
    "metrics": {
	    "train": [{"name": "psnr"}],
		"val":   [{"name": "psnr"}],
		"test":  [{"name": "psnr"}, {"name": "ssim"}],
    },
        # Metric(s) for validating and testing model. Defaults to None.
    "optimizers": [
        {
            "optimizer": {
				"name": "adam",
				"lr": 2e-4,
				"weight_decay": 0,
				"betas": [0.9, 0.99],
			},
	        "lr_scheduler": {
				"scheduler": {
					"name": "cosine_annealing_lr",
					"T_max": 400000,
					"eta_min": 1e-7,
					"last_epoch": -1
				},
				# REQUIRED: The scheduler measurement
				"interval": "epoch",
					# Unit of the scheduler's step size, could also be 'step'.
					# 'epoch' updates the scheduler on epoch end whereas 'step'
					# updates it after a optimizer update.
				"frequency": 1,
					# How many epochs/steps should pass between calls to
					# `scheduler.step()`. 1 corresponds to updating the
		            # learning rate after every epoch/step.
				"monitor": "val_loss",
					# Metric to monitor for schedulers like `ReduceLROnPlateau`
				"strict": True,
					# If set to `True`, will enforce that the value specified
					# 'monitor' is available when the scheduler is updated,
		            # thus stopping training if not found. If set to `False`,
		            # it will only produce a warning
				"name": None,
					# If using the `LearningRateMonitor` callback to monitor
		            # the learning rate progress, this keyword can be used to
		            # specify a custom logged name
			},
            "frequency": None,
        }
    ],
        # Optimizer(s) for training model. Defaults to None.
    "debug": default.debug | {
		"every_n_epochs": 5,
			# Number of epochs between debugging. To disable, set
	        # `every_n_epochs=0`. Defaults to 1.
    },
        # Debug configs. Defaults to None.
	"verbose": True,
		# Verbosity. Defaults to True.
}


# H1: - Trainer ----------------------------------------------------------------

callbacks = [
    default.model_checkpoint | {
	    "monitor": "checkpoint/psnr/train_epoch",
		    # Quantity to monitor. Defaults to None which saves a checkpoint
	        # only for the last epoch.
		"mode": "max",
			# One of {min, max}. If `save_top_k != 0`, the decision to
	        # overwrite the current save file is made based on either the
	        # maximization or the minimization of the monitored quantity.
	        # For `val_acc`, this should be `max`, for `val_loss` this should
	        # be `min`, etc.
	},
	default.learning_rate_monitor,
	default.rich_model_summary,
	default.rich_progress_bar,
]

logger = {
	"tensorboard": default.tensorboard,
}

trainer = default.trainer | {
	"default_root_dir": root,
        # Default path for logs and weights when no logger/ckpt_callback passed.
        # Can be remote file paths such as `s3://mybucket/path` or
        # 'hdfs://path/'. Defaults to os.getcwd().
}
