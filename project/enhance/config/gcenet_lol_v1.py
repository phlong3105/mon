#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GCENet model trained on LOL-v1 dataset."""

from __future__ import annotations

import mon
from mon import albumentation as A
from mon.config import default

_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
_root_dir     = _current_file.parents[1]


# region Basic

model_name = "gcenet"
data_name  = "lol_v1"
root       = _root_dir / "run"
fullname   = f"{model_name}_{data_name}"
image_size = [512, 512]
seed	   = 100
verbose    = True

# endregion


# region Model

model = {
	"name"       : model_name,     # The model's name.
	"root"       : root,           # The root directory of the model.
	"fullname"   : fullname,       # A full model name to save the checkpoint or weight.
	"channels"   : 3,              # The first layer's input channel.
	"num_classes": None,           # A number of classes, which is also the last layer's output channels.
	"classlabels": None,           # A :class:`mon.nn.data.label.ClassLabels` object that contains all labels in the dataset.
	"weights"    : None,           # The model's weights.
	"metrics"    : {
		"train": None,  # [{"name": "psnr"}],
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : None,  # [{"name": "psnr"}],
    },          # A list metrics for validating and testing model.
	"optimizers" : [
		{
            "optimizer"   : {
	            "name"        : "adam",
	            "lr"          : 0.00005,
	            "weight_decay": 0.00001,
	            "betas"       : [0.9, 0.99],
			},
			"lr_scheduler": None,
        }
    ],          # Optimizer(s) for training model.
	"verbose"    : verbose,        # Verbosity.
}

# endregion


# region Data

datamodule = {
    "name"      : data_name,
    "root"      : mon.DATA_DIR / "llie",  # A root directory where the data is stored.
	"transform" : A.Compose(transforms=[
		A.Resize(width=image_size[0], height=image_size[1]),
		# A.Flip(),
		# A.Rotate(),
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
		default.model_checkpoint | {
		    "monitor": "val/psnr",  # Quantity to monitor.
			"mode"   : "max",       # ``'min'`` or ``'max'``.
		},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir" : root,  # Default path for logs and weights.
	"gradient_clip_val": 0.1,
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
