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

model_name = "d2ce_edge"
data_name  = "ulol"
root       = root_dir / "run"
fullname   = f"{model_name}_{data_name}"
image_size = [504, 504]
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
	"num_channels": 32,		        # The number of input and output channels for subsequent layers.
	"num_iters"   : 15,             # The number of progressive loop.
	"de_encoder"  : "vits",         # The encoder for DepthAnythingV2.
	"dba_eps"     : 0.05,		    # The epsilon for DepthBoundaryAware.
	"gf_radius"   : 3,              # The radius for GuidedFilter.
	"gf_eps"	  : 1e-4,           # The epsilon for GuidedFilter.
	"bam_gamma"	  : 2.6,            # The gamma for BrightnessAttentionMap.
	"bam_ksize"   : 9,			    # The kernel size for BrightnessAttentionMap.
	"weights"     : None,           # The model's weights.
	"metrics"     : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"  : [
		{
            "optimizer"          : {
	            "name"        : "adam",
	            "lr"          : 0.00005,
	            "weight_decay": 0.00001,
	            "betas"       : [0.9, 0.99],
			},
			"lr_scheduler"       : None,
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
		# A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_AREA),
		A.ResizeMultipleOf(
			height            = image_size[0],
			width             = image_size[1],
			keep_aspect_ratio = False,
			multiple_of       = 14,
			resize_method     = "lower_bound",
			interpolation     = cv2.INTER_AREA,
		),
		# A.NormalizeImageMeanStd(
		# 	mean = [0.485, 0.456, 0.406],
		# 	std  = [0.229, 0.224, 0.225]
		# ),
		# A.Flip(),
		# A.Rotate(),
	]),  # Transformations performing on both the input and target.
    "to_tensor" : True,          # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,         # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 4,             # The number of samples in one forward pass.
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
	"gradient_clip_val": 0.1,
	"log_image_every_n_epochs": 1,
	"logger"           : {
		"tensorboard": default.tensorboard,
	},
	"max_epochs"       : 50,
	"strategy"         : "ddp_find_unused_parameters_true",
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,   # Default path for saving results.
}

# endregion
