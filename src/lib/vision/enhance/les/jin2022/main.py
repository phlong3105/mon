#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from enhancenet import EnhanceNet
from mon import RUN_DIR
from utils import *

console = mon.console


def main():
    args = parse_args()
    if args is None:
        exit()

    #
    gan = EnhanceNet(args)
    gan.build_model()

    if args.phase == "test" :
        gan.test()
        console.log("Test finished!")


def check_args(args):
    check_folder(os.path.join(args.result_dir, args.dataset, "model"))
    try:
        assert args.epoch >= 1
    except:
        console.log("Number of epochs must be larger than or equal to one")
    try:
        assert args.batch_size >= 1
    except:
        console.log("Batch size must be larger than or equal to one")
    return args


def parse_args():
    desc   = "Pytorch implementation of NightImageEnhancement"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--data",               type=str,      default="LOL",     help="Dataset path")
    parser.add_argument("--data-name",          type=str,      default="LOL",     help="Dataset_name")
    parser.add_argument("--phase",              type=str,      default="test",    help="[train / test]")
    parser.add_argument("--iteration",          type=int,      default=900000,    help="The number of training iterations")
    parser.add_argument("--batch_size",         type=int,      default=1,         help="The size of batch size")
    parser.add_argument("--image-size",         type=int,      default=512,       help="The training size of image")
    parser.add_argument("--input-channels",     type=int,      default=3,         help="The size of image channel")
    parser.add_argument("--channels",           type=int,      default=64,        help="base channel number per layer")
    parser.add_argument("--n-res",              type=int,      default=4,         help="The number of resblock")
    parser.add_argument("--n-dis",              type=int,      default=6,         help="The number of discriminator layer")
    parser.add_argument("--adv-weight",         type=int,      default=1,         help="Weight for GAN Loss")
    parser.add_argument("--atten-weight",       type=int,      default=0.5,       help="Weight for Attention Loss")
    parser.add_argument("--identity-weight",    type=int,      default=5,         help="Weight for Identity Loss")
    parser.add_argument("--use-gray-feat-loss", type=str2bool, default=True,      help="use Structure and HF-Features Consistency Losses")
    parser.add_argument("--feat-weight",        type=int,      default=1,         help="Weight for Structure and HF-Features Consistency Losses")
    parser.add_argument("--lr",                 type=float,    default=0.0001,    help="The learning rate")
    parser.add_argument("--weight-decay",       type=float,    default=0.0001,    help="The weight decay")
    parser.add_argument("--decay-flag",         type=str2bool, default=True,      help="The decay flag")
    parser.add_argument("--output-dir",         type=str,      default=RUN_DIR / "predict/zerodce", help="Directory name to save the results")
    parser.add_argument("--print-freq",         type=int,      default=1000,      help="The number of image print freq")
    parser.add_argument("--save-freq",          type=int,      default=100000,    help="The number of model save freq")
    parser.add_argument("--device",             type=str,      default="cuda",    choices=["cpu", "cuda"], help="Set gpu mode; [cpu, cuda]")
    parser.add_argument("--benchmark-flag",     type=str2bool, default=False)
    parser.add_argument("--resume",             type=str2bool, default=True)
    return check_args(parser.parse_args())


if __name__ == "__main__":
    main()
