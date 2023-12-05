import argparse
import os
from glob import glob

import numpy as np

import mon
from model import RetinexNet
from mon import DATA_DIR, RUN_DIR

console = mon.console


def train(model):

    lr = args.lr * np.ones([args.epochs])
    lr[20:] = lr[0] / 10.0

    train_low_data_names  = glob(str(args.data_low)  + "/*")
    train_low_data_names.sort()
    train_high_data_names = glob(str(args.data_high) + "/*")
    train_high_data_names.sort()
    eval_low_data_names   = glob(str(args.data) + "/eval/low/*.*")
    eval_low_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    console.log("Number of training data: %d" % len(train_low_data_names))

    model.train(
        train_low_data_names,
        train_high_data_names,
        eval_low_data_names,
        batch_size       = args.batch_size,
        patch_size       = args.patch_size,
        epoch            = args.epochs,
        lr               = lr,
        vis_dir          = args.visual_dir,
        ckpt_dir         = args.checkpoint_dir,
        eval_every_epoch = 10,
        train_phase      = "Decom"
    )

    model.train(
        train_low_data_names,
        train_high_data_names,
        eval_low_data_names,
        batch_size       = args.batch_size,
        patch_size       = args.patch_size,
        epoch            = args.epochs,
        lr               = lr,
        vis_dir          = args.visual_dir,
        ckpt_dir         = args.checkpoint_dir,
        eval_every_epoch = 10,
        train_phase      = "Relight"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-low",                  default=DATA_DIR, help="directory storing the training data")
    parser.add_argument("--data-high",                 default=DATA_DIR, help="directory storing the training data")
    parser.add_argument("--gpu",                       default="0",   help="GPU ID (-1 for CPU)")
    parser.add_argument("--epochs",        type=int,   default=100,   help="number of total epochs")
    parser.add_argument("--batch_size",    type=int,   default=16,    help="number of samples in one batch")
    parser.add_argument("--patch_size",    type=int,   default=96,    help="patch size")
    parser.add_argument("--lr",            type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--checkpoint-dir",            default=RUN_DIR / "train/vision/enhance/llie/retinexnet", help="directory for checkpoints")
    args = parser.parse_args()

    args.data_low  = mon.Path(args.data_low)
    args.data_high = mon.Path(args.data_high)

    if args.gpu != "-1":
        # Create directories for saving the checkpoints and visuals
        args.visual_dir = args.checkpoint_dir + "/visuals/"
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.exists(args.visual_dir):
            os.makedirs(args.visual_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # Create the model
        model = RetinexNet().cuda()
        # Train the model
        train(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
