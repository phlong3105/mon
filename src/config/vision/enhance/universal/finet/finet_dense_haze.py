#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FINet model trained on Dense-Haze dataset."""

from __future__ import annotations

import albumentations as A

from config import default
from mon import DATA_DIR, RUN_DIR

# region Basic

root         = RUN_DIR / "train"
project      = "finet"
model_name   = "finet"
model_config = None
data_name    = "dense-haze"
num_classes  = None
fullname     = f"{model_name}-{data_name}"
image_size   = [256, 256]
seed	     = 100
verbose 	 = True

# endregion


# region Model

model = {
	"config"     : model_config,   # The model's configuration that is used to build the model.
	"hparams"    : None,           # Model's hyperparameters.
	"channels"   : 3,              # The first layer's input channel.
	"num_classes": None,           # A number of classes, which is also the last layer's output channels.
	"classlabels": None,           # A :class:`mon.nn.data.label.ClassLabels` object that contains all labels in the dataset.
	"weights"    : None,           # The model's weights.
	"name"       : model_name,     # The model's name.
	"variant"    : None,           # The model's variant.
	"fullname"   : fullname,       # A full model name to save the checkpoint or weight.
	"root"       : root,           # The root directory of the model.
	"project"    : project,        # A project name.
	"phase"      : "training",     # The model's running phase.
	"loss"       : {
		"name": "psnr_loss", "loss_weight": 0.5, "to_y": True,
	},          # Loss function for training the model.
	"metrics"    : {
		"train": None,  # [{"name": "psnr"}],
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : None,  # [{"name": "psnr"}],
    },          # A list metrics for validating and testing model.
	"optimizers" : [
		{
            "optimizer"   : {
	            "name"        : "adam",
	            "lr"          : 2e-4,
	            "weight_decay": 0,
	            "betas"       : [0.9, 0.99],
			},
	        "lr_scheduler": {
				"scheduler": {
					"name"      : "cosine_annealing_lr",
					"T_max"     : 400000,
					"eta_min"   : 1e-7,
					"last_epoch": -1
				},
				# REQUIRED: The scheduler measurement
				"interval" : "epoch",     # Unit of the scheduler's step size. One of ['step', 'epoch'].
				"frequency": 1,           # How many epochs/steps should pass between calls to `scheduler.step()`.
				"monitor"  : "val/loss",  # Metric to monitor for schedulers like `ReduceLROnPlateau`.
				"strict"   : True,
				"name"     : None,
			},
        }
    ],          # Optimizer(s) for training model.
	"debug"      : default.debug,  # Debug configs.
	"verbose"    : verbose,        # Verbosity.
}

# endregion


# region Data

datamodule = {
    "name"        : data_name,
    "root"        : DATA_DIR / "dehaze",  # A root directory where the data is stored.
    "image_size"  : image_size,   # The desired image size in HW format.
    "transform"   : A.Compose([
        A.Resize(width=image_size[0], height=image_size[1]),
		A.Flip(),
		A.Rotate(),
    ]),  # Transformations performing on both the input and target.
    "to_tensor"   : True,         # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data"  : False,        # If ``True``, cache data to disk for faster loading next time.
    "cache_images": False,        # If ``True``, cache images into memory for faster training.
    "batch_size"  : 16,            # The number of samples in one forward pass.
    "devices"     : 0,            # A list of devices to use. Default: ``0``.
    "shuffle"     : True,         # If ``True``, reshuffle the datapoints at the beginning of every epoch.
    "verbose"     : verbose,      # Verbosity.
}

# endregion


# region Training

trainer = default.trainer | {
	"callbacks"        : [
		default.model_checkpoint | {
		    "monitor": "val/psnr",  # Quantity to monitor.
			"mode"   : "max",       # ``'min'`` or ``'max'``.
		},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir" : root,  # Default path for logs and weights.
	"gradient_clip_val": 0.01,
	"logger"           : {
		"tensorboard": default.tensorboard,
	},
	
}

# endregion


# region Predicting

predictor = default.predictor | {}

# endregion
