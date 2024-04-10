#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HINet model trained on Rain13K dataset."""

from __future__ import annotations

import mon
from mon import albumentation as A
from mon.config import default

_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
_root_dir     = _current_file.parents[1]


# region Basic

model_name  = "hinet"
data_name   = "rain13k"
root        = _root_dir / "run"
fullname    = f"{model_name}_{data_name}"
image_size  = [256, 256]
seed	    = 100
verbose     = True

# endregion


# region Model

model = {
	"name"        : model_name,     # The model's name.
	"root"        : root,           # The root directory of the model.
	"fullname"    : fullname,       # A full model name to save the checkpoint or weight.
	"in_channels" : 3,              # The first layer's input channel.
	"out_channels": None,           # A number of classes, which is also the last layer's output channels.
	"num_channels": 64,
	"depth"       : 5,
	"relu_slope"  : 0.2,
	"in_pos_left" : 0,
	"in_pos_right": 4,
	"weights"     : None,           # The model's weights.
	"loss"        : {
		"name"       : "psnr_loss",
		"loss_weight": 0.5,
		"to_y"       : True,
		# "name": "l1_loss",
	},          # Loss function for training the model.
	"metrics"     : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"  : [
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
				"interval" : "epoch",       # Unit of the scheduler's step size. One of ['step', 'epoch'].
				"frequency": 1,             # How many epochs/steps should pass between calls to `scheduler.step()`.
				"monitor"  : "train/loss",  # Metric to monitor for schedulers like `ReduceLROnPlateau`.
				"strict"   : True,
				"name"     : None,
			},
        }
    ],          # Optimizer(s) for training model.
	"verbose"     : verbose,        # Verbosity.
}

# endregion


# region Data

datamodule = {
    "name"      : data_name,
    "root"      : mon.DATA_DIR / "derain",  # A root directory where the data is stored.
    "transform" : A.Compose(transforms=[
		A.Resize(width=image_size[0], height=image_size[1]),
		A.Flip(),
		A.Rotate(),
    ]),  # Transformations performing on both the input and target.
    "to_tensor" : True,         # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,        # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 32,           # The number of samples in one forward pass.
    "devices"   : 0,            # A list of devices to use. Default: ``0``.
    "shuffle"   : True,         # If ``True``, reshuffle the datapoints at the beginning of every epoch.
    "verbose"   : verbose,      # Verbosity.
}

# endregion


# region Training

trainer = default.trainer | {
	"callbacks"        : [
		default.log_training_progress,
		default.model_checkpoint | {"monitor": "val/psnr", "mode": "max"},
		default.model_checkpoint | {"monitor": "val/ssim", "mode": "max", "save_last": True},
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

predictor = default.predictor | {
	"default_root_dir": root,   # Default path for saving results.
}

# endregion
