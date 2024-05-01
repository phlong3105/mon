#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import mon
from mon import albumentation as A
from mon.config import default

_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
_root_dir     = _current_file.parents[1]


# region Basic

model_name = "hvi_cidnet"
data_name  = "lol_v1"
root       = _root_dir / "run"
fullname   = f"{model_name}_{data_name}"
image_size = [256, 256]
seed	   = 1000000
verbose    = True

# endregion


# region Model

model = {
	"name"        : model_name,     # The model's name.
	"root"        : root,           # The root directory of the model.
	"fullname"    : fullname,       # A full model name to save the checkpoint or weight.
	"in_channels" : 3,              # The first layer's input channel.
	"out_channels": None,           # A number of classes, which is also the last layer's output channels.
	"channels"    : [36, 36, 72, 144],
	"heads"       : [1, 2, 4, 8],
	"norm"        : False,
	"hvi_weight"  : 1.0,
	"weights"     : None,           # The model's weights.
	"loss"        : None,           # Loss function for training the model.
	"metrics"     : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"  : [
		{
            "optimizer"          : {
				"name": "adam",
				"lr"  : 0.0001,
			},
	        "lr_scheduler"       : {
				"scheduler": {
					"name"           : "cosine_annealing_restart_lr",
					"periods"        : [1000],
					"restart_weights": [1],
					"eta_min"        : 1e-7,
				},
				"interval" : "epoch",       # Unit of the scheduler's step size. One of ['step', 'epoch'].
				"frequency": 1,             # How many epochs/steps should pass between calls to `scheduler.step()`.
				"monitor"  : "train/loss",  # Metric to monitor for schedulers like `ReduceLROnPlateau`.
				"strict"   : True,
				"name"     : None,
			},
			"network_params_only": True,
        }
    ],          # Optimizer(s) for training model.
	"verbose"     : verbose,        # Verbosity.
}

# endregion


# region Data

datamodule = {
    "name"      : data_name,
    "root"      : mon.DATA_DIR / "llie",  # A root directory where the data is stored.
    "transform" : A.Compose(transforms=[
		A.RandomCrop(height=image_size[0], width=image_size[1]),
	    A.HorizontalFlip(),
		A.VerticalFlip(),
    ]),  # Transformations performing on both the input and target.
    "to_tensor" : True,         # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,        # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 8,            # The number of samples in one forward pass.
    "devices"   : 0,            # A list of devices to use. Default: ``0``.
    "shuffle"   : True,         # If ``True``, reshuffle the datapoints at the beginning of every epoch.
    "verbose"   : verbose,      # Verbosity.
}

# endregion


# region Training

trainer = default.trainer | {
	"callbacks"       : [
		default.log_training_progress,
		default.model_checkpoint | {"monitor": "val/psnr", "mode": "max"},
		default.model_checkpoint | {"monitor": "val/ssim", "mode": "max", "save_last": True},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir": root,  # Default path for logs and weights.
	"logger"          : {
		"tensorboard": default.tensorboard,
	},
	"max_epochs"      : 1000,
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,   # Default path for saving results.
}

# endregion
