#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from enhancenet import EnhanceNet
from mon import RUN_DIR, ZOO_DIR, DATA_DIR
from utils import *

console = mon.console


def predict(args: argparse.Namespace):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Data: {args.input_dir}")

    #
    gan = EnhanceNet(args)
    gan.build_model()
    gan.load(weights=args.weights)

    # Measure efficiency score
    if args.benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = gan.genA2B,
            image_size = args.image_size,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")

    #
    with torch.no_grad():
        image_paths = list(args.input_dir.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                enhanced_image, run_time = gan.predict(image_path=image_path)
                output_path = args.output_dir / image_path.name
                cv2.imwrite(str(output_path), enhanced_image)
                sum_time += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")


def parse_args():
    desc   = "Pytorch implementation of NightImageEnhancement"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input-dir",          type=str,      default=DATA_DIR, help="Dataset path")
    parser.add_argument("--output-dir",         type=str,      default=RUN_DIR / "predict/vision/enhance/les/jin2022", help="Directory name to save the results")
    parser.add_argument("--data-name",          type=str,      default="LOL",    help="Dataset_name")
    parser.add_argument("--phase",              type=str,      default="test",   help="[train / test]")
    parser.add_argument("--weights",            type=str,      default=ZOO_DIR / "vision/enhance/les/jin2022/delighteffects_params_0600000.pt")
    parser.add_argument("--iteration",          type=int,      default=900000,   help="The number of training iterations")
    parser.add_argument("--batch_size",         type=int,      default=1,        help="The size of batch size")
    parser.add_argument("--image-size",         type=int,      default=512,      help="The training size of image")
    parser.add_argument("--input-channels",     type=int,      default=3,        help="The size of image channel")
    parser.add_argument("--channels",           type=int,      default=64,       help="base channel number per layer")
    parser.add_argument("--n-res",              type=int,      default=4,        help="The number of resblock")
    parser.add_argument("--n-dis",              type=int,      default=6,        help="The number of discriminator layer")
    parser.add_argument("--adv-weight",         type=int,      default=1,        help="Weight for GAN Loss")
    parser.add_argument("--atten-weight",       type=int,      default=0.5,      help="Weight for Attention Loss")
    parser.add_argument("--identity-weight",    type=int,      default=5,        help="Weight for Identity Loss")
    parser.add_argument("--use-gray-feat-loss", type=str2bool, default=True,     help="use Structure and HF-Features Consistency Losses")
    parser.add_argument("--feat-weight",        type=int,      default=1,        help="Weight for Structure and HF-Features Consistency Losses")
    parser.add_argument("--lr",                 type=float,    default=0.0001,   help="The learning rate")
    parser.add_argument("--weight-decay",       type=float,    default=0.0001,   help="The weight decay")
    parser.add_argument("--decay-flag",         type=str2bool, default=True,     help="The decay flag")
    parser.add_argument("--print-freq",         type=int,      default=1000,     help="The number of image print freq")
    parser.add_argument("--save-freq",          type=int,      default=100000,   help="The number of model save freq")
    parser.add_argument("--device",             type=str,      default="cuda",   choices=["cpu", "cuda"], help="Set gpu mode; [cpu, cuda]")
    parser.add_argument("--benchmark-flag",     type=str2bool, default=False)
    parser.add_argument("--resume",             type=str2bool, default=True)
    parser.add_argument("--benchmark",          action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()
    predict(args)
