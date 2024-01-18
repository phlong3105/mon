#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/KarelZhang/RUAS

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from torch.autograd import Variable

import mon
# noinspection PyUnresolvedReferences
import utils
from model import Network
from mon import ZOO_DIR, DATA_DIR, RUN_DIR
from multi_read_data import MemoryFriendlyLoader

console = mon.console


def train(args):
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    if not torch.cuda.is_available():
        console.log("No gpu device available.")
        sys.exit(1)
    
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled   = True
    torch.cuda.manual_seed(args.seed)
    logging.info("GPU device = %d" % args.gpu)
    logging.info("args = %s", args)

    model = Network()
    model = model.cuda()
    if args.load_pretrain:
        model_dict = torch.load(str(args.weights))
        model.load_state_dict(model_dict)
    
    # Prepare DataLoader
    # train_low_data_names = r"D:\ZJA\data\LOL\OR\trainA/*.png"
    # train_low_data_names = r"H:\image-enhance\UPE500\OR\trainA/*.png"
    train_low_data_names = args.input_dir + "/*"
    train_dataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        pin_memory  = True,
        num_workers = 0,
    )
    model.train()
    
    total_step = 0
    with mon.get_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow] Training"
        ):
            for iteration, img_lowlight in enumerate(train_loader):
                total_step      += 1
                input            = Variable(img_lowlight, requires_grad=False).cuda()
                loss1, loss2, _  = model.optimizer(input, input, total_step)
                
                if total_step % args.report_freq == 0 and total_step != 0:
                    torch.save(model.state_dict(), args.checkpoints_dir / "best.pt")
   
    """
    total_step = 0
    while total_step < 800:
        input = next(iter(train_queue))
        total_step += 1
        model.train()
        input = Variable(input, requires_grad=False).cuda()
        loss1, loss2, _ = model.optimizer(input, input, total_step)

        if total_step % args.report_freq == 0 and total_step != 0:
            # console.log("train %03d %f %f", total_step, loss1, loss2)
            utils.save(model, os.path.join(model_path, "best.pt"))
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ruas")
    parser.add_argument("--input-dir",       type=str,   default=DATA_DIR)
    parser.add_argument("--weights",         type=str,   default=ZOO_DIR / "vision/enhance/llie/ruas/ruas-lol.pt")
    parser.add_argument("--load-pretrain",   type=bool,  default=False)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch-size",      type=int,   default=1,  help="Batch size")
    parser.add_argument("--report-freq",     type=float, default=50, help="Report frequency")
    parser.add_argument("--gpu",             type=int,   default=0,  help="GPU device id")
    parser.add_argument("--seed",            type=int,   default=2,  help="Random seed")
    parser.add_argument("--checkpoints-dir", type=str,   default=RUN_DIR / "train/vision/enhance/llie/ruas")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
    
    """
    EXP_path = r"./EXP\train/"
    if not os.path.isdir(EXP_path):
        os.mkdir(EXP_path)
    model_path = EXP_path + "\model/"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream  = sys.stdout,
        level   = logging.INFO,
        format  = log_format,
        datefmt = "%m/%d %I:%M:%S %p"
    )
    fh = logging.FileHandler(os.path.join(EXP_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    train(args)
    """
