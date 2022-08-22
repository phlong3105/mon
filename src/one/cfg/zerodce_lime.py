#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zero-DCE trained on LIME dataset.
"""

from __future__ import annotations

import default
from one.constants import RUNS_DIR
from one.constants import VISION_BACKEND
from one.vision.transformation import Resize


# H1: - Basic ------------------------------------------------------------------

data_name  = "lime"
model_name = "zerodce"
model_cfg  = "zerodce.yaml"
fullname   = f"{model_name}-{data_name}"
root       = RUNS_DIR / fullname


# H1: - Data -------------------------------------------------------------------

data = {
    "name": data_name,
        # Dataset's name.
    "shape": [3, 512, 512],
        # Image shape as [C, H, W], [H, W], or [S, S].
    "transform": None,
        # Functions/transforms that takes in an input sample and returns a
        # transformed version.
    "target_transform": None,
        # Functions/transforms that takes in a target and returns a transformed
        # version.
    "transforms": [
        Resize(size=[3, 512, 512]),
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
    "batch_size": 8,
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
    "root": root,
        # The root directory of the model. Defaults to RUNS_DIR.
    "name": model_name,
        # Model's name. In case None is given, it will be
        # `self.__class__.__name__`. Defaults to None.
    "fullname": fullname,
        # Model's fullname in the following format: {name}-{data_name}-{postfix}.
        # In case None is given, it will be `self.basename`. Defaults to None.
    "cfg": model_cfg,
        # Model's layers configuration. It can be an external .yaml path or a
        # dictionary. Defaults to None means you should define each layer
        # manually in `self.parse_model()` method.
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
    "loss": None,
        # Loss function for training model. Defaults to None.
    "metrics": None,
        # Metric(s) for validating and testing model. Defaults to None.
    "optimizers": [
        {
            "optimizer": {
				"name": "adam",
				"lr": 0.0001,
				"weight_decay": 0.0001,
				"betas": [0.9, 0.99],
			},
            "frequency": None,
        }
    ],
        # Optimizer(s) for training model. Defaults to None.
    "debug": default.debug,
        # Debug configs. Defaults to None.
}


# H1: - Trainer ----------------------------------------------------------------

callbacks = [
    default.model_checkpoint | {
        "root": RUNS_DIR,
            # Root directory to save checkpoint files
	},
	default.learning_rate_monitor,
	default.rich_model_summary,
	default.rich_progress_bar,
]

loggers = {
	"tensorboard": default.tensorboard | {
		"save_dir": root,
			# Save directory.
	},
}

trainer = default.trainer
