#!/usr/bin/edenoised1nv python
# -*- coding: utf-8 -*-

from __future__ import annotations

import mon
from mon import albumentation as A
from mon.config import default

current_file = mon.Path(__file__).absolute()


# region Basic

model_name = "zero_ig_re"
data_name  = "lol_v2_real"
root       = current_file.parents[1] / "run"
data_root  = mon.DATA_DIR / "enhance" / "llie"
project    = None
variant    = None
fullname   = f"{model_name}_{data_name}"
image_size = [512, 512]
seed	   = 2
verbose    = True

# endregion


# region Model

model = {
	"name"          : model_name,     # The model's name.
	"fullname"      : fullname,       # A full model name to save the checkpoint or weight.
	"root"          : root,           # The root directory of the model.
	"in_channels"   : 3,              # The first layer's input channel.
	"out_channels"  : None,           # A number of classes, which is also the last layer's output channels.
	"num_channels"  : 64,			  # The number of input and output channels for subsequent layers.
	"embed_channels": 48,             # The number of channels in the embedding space.
	"weights"       : None,           # The model's weights.
	"metrics"       : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"    : [
		{
            "optimizer"          : {
	            "name"        : "adam",
	            "lr"          : 0.0003,
	            "weight_decay": 0.0003,
	            "betas"       : [0.9, 0.99],
			},
			"lr_scheduler"       : None,
			"network_params_only": True,
        }
    ],          # Optimizer(s) for training model.
	"debug"         : False,          # If ``True``, run the model in debug mode (when predicting).
	"verbose"       : verbose,        # Verbosity.
}

# endregion


# region Data

data = {
    "name"      : data_name,
    "root"      : data_root,     # A root directory where the data is stored.
	"transform" : A.Compose(transforms=[
		A.Resize(height=image_size[0], width=image_size[1]),
		# A.Flip(),
		# A.Rotate(),
	]),  # Transformations performing on both the input and target.
    "to_tensor" : True,          # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,         # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 1,             # The number of samples in one forward pass.
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
			"mode"    : "max",
		},
		default.model_checkpoint | {
			"filename": fullname,
			"monitor" : "val/ssim",
			"mode"    : "max",
		},
		default.model_checkpoint | {
			"filename" : fullname,
			"monitor"  : "train/loss",
			"mode"     : "min",
			"save_last": True,
		},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir" : root,  # Default path for logs and weights.
	"gradient_clip_val": None,
	"log_image_every_n_epochs": 500,
	"logger"           : {
		"tensorboard": default.tensorboard,
	},
	"max_epochs"       : 2000,
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,  # Default path for saving results.
}

# endregion
