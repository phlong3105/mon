#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/vis-opt-group/SCI

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image
from torch.autograd import Variable

import mon
import utils
from model import *
from mon import ZOO_DIR
from multi_read_data import MemoryFriendlyLoader

console = mon.console


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, "png")


def train(args):
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.input_dir}")
    
    if not torch.cuda.is_available():
        console.log("No gpu device available")
        sys.exit(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    
    utils.create_exp_dir(args.checkpoint_dir, scripts_to_save=glob.glob("*.py"))
    image_path = args.checkpoint_dir / "image_epochs"
    
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        if not args.cuda:
            torch.set_default_tensor_type("torch.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled   = True
    torch.cuda.manual_seed(args.seed)
    console.log("gpu device = %s" % args.gpu)
    console.log("args = %s", args)
    
    model = Network(stage=args.stage)
    model.enhance.in_conv.apply(model.weights_init)
    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    model.calibrate.in_conv.apply(model.weights_init)
    model.calibrate.convs.apply(model.weights_init)
    model.calibrate.out_conv.apply(model.weights_init)
    model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB        = utils.count_parameters_in_MB(model)
    console.log("model size = %f", MB)
    console.log(MB)

    train_low_data_names = str(args.input_dir)
    train_dataset        = MemoryFriendlyLoader(img_dir=train_low_data_names, task="train")
    test_low_data_names  = "./data/medium"
    test_dataset         = MemoryFriendlyLoader(img_dir=test_low_data_names, task="test")
    train_queue = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        pin_memory  = True,
        num_workers = 0,
        shuffle     = True
    )
    test_queue = torch.utils.data.DataLoader(
        test_dataset,
        batch_size  = 1,
        pin_memory  = True,
        num_workers = 0,
        shuffle     = True
    )

    total_step = 0
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow] Training"
        ):
            model.train()
            losses = []
            for batch_idx, (input, _) in enumerate(train_queue):
                total_step += 1
                input       = Variable(input, requires_grad=False).cuda()
    
                optimizer.zero_grad()
                loss = model._loss(input)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
    
                losses.append(loss.item())
                console.log("train-epoch %03d %03d %f", epoch, batch_idx, loss)
    
            console.log("train-epoch %03d %f", epoch, np.average(losses))
            utils.save(model, args.checkpoint_dir / "last.pt")
    
            if epoch % 1 == 0 and total_step != 0:
                console.log("train %03d %f", epoch, loss)
                model.eval()
                with torch.no_grad():
                    for _, (input, image_name) in enumerate(test_queue):
                        input       = Variable(input, volatile=True).cuda()
                        illu_list, ref_list, input_list, atten = model(input)
                        image_name  = image_name[0].split("\\")[-1].split(".")[0]
                        output_path = image_path / f"{image_name}_{epoch}.png"
                        save_images(ref_list[0], str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SCI")
    parser.add_argument("--input-dir",       type=str,   default="data/train_data/")
    parser.add_argument("--weights",         type=str,   default=ZOO_DIR / "vision/enhance/llie/sci/sci-medium.pt")
    parser.add_argument("--load-pretrain",   type=bool,  default=False)
    parser.add_argument("--batch-size",      type=int,   default=1,      help="batch size")
    parser.add_argument("--epochs",          type=int,   default=1000,   help="epochs")
    parser.add_argument("--lr",              type=float, default=0.0003, help="learning rate")
    parser.add_argument("--stage",           type=int,   default=3,      help="epochs")
    parser.add_argument("--cuda",            type=bool,  default=True,   help="Use CUDA to train model")
    parser.add_argument("--gpu",             type=str,   default="0",    help="gpu device id")
    parser.add_argument("--seed",            type=int,   default=2,      help="random seed")
    parser.add_argument("--checkpoints-dir", type=str,   default=mon.RUN_DIR/"train/iat", help="location of the data corpus")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
