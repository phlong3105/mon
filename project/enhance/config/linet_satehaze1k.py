#!/usr/bin/edenoised1nv python
# -*- coding: utf-8 -*-

from __future__ import annotations

import cv2

import mon
from mon import albumentation as A
from mon.config import default

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]


# region Basic

model_name = "linet"
data_name  = "satehaze1k"
root       = root_dir / "run"
fullname   = f"{model_name}_{data_name}"
image_size = [512, 512]
seed	   = 100
verbose    = True

# endregion


# region Model

model = {
	"name"        : model_name,     # The model's name.
	"root"        : root,           # The root directory of the model.
	"fullname"    : fullname,       # A full model name to save the checkpoint or weight.
	"in_channels" : 3,              # The first layer's input channel.
	"out_channels": None,           # A number of classes, which is also the last layer's output channels.
	"num_channels": 64,		        # The number of input and output channels for subsequent layers.
	"depth" 	  : 5,              # The number of blocks in the model.
	"relu_slope"  : 0.2,            # The slope of the Leaky ReLU activation function.
	"in_pos_left" : 0,		        # The layer index to begin applying the Instance Normalization.
	"in_pos_right": 4,              # The layer index to end applying the Instance Normalization
	"r"           : 0.5,            # The initial probability of applying the Instance Normalization.
	"eps"	      : 1e-5,           # The epsilon value for the Instance Normalization.
	"weights"     : None,           # The model's weights.
	"loss"		 : {
		"name"       : "l1_loss",
		# "loss_weight": 0.5,
		# "to_y"       : True,
	},
	"metrics"     : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"  : [
		{
            "optimizer"          : {
	            "name"        : "adam",
	            "lr"          : 0.0002,
	            "weight_decay": 0,
	            "betas"       : [0.9, 0.99],
			},
			"lr_scheduler"       : {
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
			"network_params_only": True,
        }
    ],          # Optimizer(s) for training model.
	"verbose"     : verbose,        # Verbosity.
}

# endregion


# region Data

datamodule = {
    "name"      : data_name,
    "root"      : mon.DATA_DIR / "dehaze",  # A root directory where the data is stored.
	"transform" : A.Compose(transforms=[
		A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_AREA),
		A.Flip(p=0.5),
		A.Rotate(p=0.5),
	]),  # Transformations performing on both the input and target.
    "to_tensor" : True,          # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,         # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 8,             # The number of samples in one forward pass.
    "devices"   : 0,             # A list of devices to use. Default: ``0``.
    "shuffle"   : True,          # If ``True``, reshuffle the datapoints at the beginning of every epoch.
    "verbose"   : verbose,       # Verbosity.
}

# endregion


# region Training

trainer = default.trainer | {
	"callbacks"        : [
		default.log_training_progress,
		default.model_checkpoint | {
			"filename": fullname,
			"monitor" : "val/psnr",
			"mode"    : "max"
		},
		default.model_checkpoint | {
			"filename" : fullname,
			"monitor"  : "val/ssim",
			"mode"     : "max",
			"save_last": True,
		},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir" : root,  # Default path for logs and weights.
	"gradient_clip_val": 0.01,
	"log_image_every_n_epochs": 1,
	"logger"           : {
		"tensorboard": default.tensorboard,
	},
	"max_epochs"       : 200,
	"strategy"         : "ddp_find_unused_parameters_true",
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,   # Default path for saving results.
}

# endregion
