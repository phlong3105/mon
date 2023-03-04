#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet model trained on CIFAR10 dataset. This is the 'Hello World'
deep learning models.
"""

from __future__ import annotations

import albumentations as A

from mon.createml.config import default
from mon.globals import RUN_DIR

# region Basic

root         = RUN_DIR / "train"
project      = "alexnet"
model_name   = "alexnet"
model_config = "alexnet.yaml"
data_name    = "cifar10"
num_classes  = 10
fullname     = f"{model_name}-{data_name}"
image_size   = [64, 64]

# endregion


# region Model

model = {
	"config"     : model_config,   # The model's configuration that is used to build the model.
	"hparams"    : None,           # Model's hyperparameters.
	"channels"   : 3,              # The first layer's input channel.
	"num_classes": num_classes,    # A number of classes, which is also the last layer's output channels.
	"classlabels": None,           # A :class:`mon.coreml.data.label.ClassLabels` object that contains all labels in the dataset.
	"weights"    : None,           # The model's weights.
	"name"       : model_name,     # The model's name.
	"variant"    : None,           # The model's variant.
	"fullname"   : fullname,       # A full model name to save the checkpoint or weight.
	"root"       : root,           # The root directory of the model.
	"project"    : project,        # A project name.
	"phase"      : "training",     # The model's running phase.
	"loss"       : {
		"name": "cross_entropy_loss",
	},          # Loss function for training the model.
	"metrics"    : {
	    "train": [{"name": "accuracy", "num_classes": num_classes}],
		"val"  : [{"name": "accuracy", "num_classes": num_classes}],
		"test" : [{"name": "accuracy", "num_classes": num_classes}],
    },          # A list metrics for validating and testing model.
	"optimizers" : [
		{
            "optimizer"   : {
	            "name"        : "adam",
	            "lr"          : 0.0001,
	            "weight_decay": 0.0001,
	            "betas"       : [0.9, 0.99],
			},
	        "lr_scheduler": None,
            "frequency"   : None,
        }
    ],          # Optimizer(s) for training model.
	"debug"      : default.debug,  # Debug configs.
	"verbose"    : True,           # Verbosity.
}

# endregion


# region Data

datamodule = {
    "name"        : data_name,    # A root directory where the data is stored.
    # "root"        : DATA_DIR / "a2i2-haze",
    "image_size"  : image_size,   # The desired image size in HW format.
    "transform"   : A.Compose([
        A.Resize(width=image_size[0], height=image_size[1]),
    ]),  # Transformations performing on both the input and target.
    "to_tensor"   : False,        # If True, convert input and target to :class:`torch.Tensor`.
    "cache_data"  : False,        # If True, cache data to disk for faster loading next time.
    "cache_images": False,        # If True, cache images into memory for faster training.
    "batch_size"  : 8,            # The number of samples in one forward pass.
    "devices"     : 0,            # A list of devices to use. Defaults to 0.
    "shuffle"     : True,         # If True, reshuffle the datapoints at the beginning of every epoch.
    "verbose"     : True,         # Verbosity.
}

# endregion


# region Training

trainer = default.trainer | {
	"callbacks"       : [
		default.model_checkpoint | {
		    "monitor": "checkpoint/psnr/train_epoch",  # Quantity to monitor.
			"mode"   : "max",                          # 'min' or 'max'.
		},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir": root,  # Default path for logs and weights.
	"logger"          : {
		"tensorboard": default.tensorboard,
	},
	
}

# endregion
