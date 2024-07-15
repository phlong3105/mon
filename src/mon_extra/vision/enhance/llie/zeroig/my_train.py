#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
from PIL import Image
from torch.autograd import Variable

import mon
import utils
from model import *
from multi_read_data import DataLoader

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def train(args: argparse.Namespace):
    # General config
    fullname = args.fullname
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    device   = mon.set_device(args.device)
    epochs   = args.epochs
    seed     = args.seed
    verbose  = args.verbose
    mon.set_random_seed(seed)
    
    # Directory
    weights_dir = save_dir
    debug_dir   = save_dir / "debug"
    weights_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Log
    log_file   = save_dir / "log.txt"
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("train file name = %s", os.path.split(__file__))
    logging.info("gpu device = %s" % device)
    logging.info("args = %s", args)
    
    # Model
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True
        cudnn.enabled   = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        logging.info('no gpu device available')
        sys.exit(1)
    
    model = Network()
    if weights is not None and mon.Path(weights).is_weights_file():
        model.load_state_dict(torch.load(weights))
    # utils.save(model, str(weights_dir / "initial_weights.pt"))
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)
    model = model.to(device)
    MB    = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    console.log(MB)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    
    # Data I/O
    data          = mon.DATA_DIR / args.data_dir
    train_dataset = DataLoader(img_dir=data, task="train")
    test_dataset  = DataLoader(img_dir=data, task="test")
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        generator   = torch.Generator(device="cuda")
    )
    test_loader   = torch.utils.data.DataLoader(
        test_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
        generator   = torch.Generator(device="cuda")
    )
    
    # Training
    model.train()
    total_step = 0
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            losses = []
            for idx, (input, img_name) in enumerate(train_loader):
                total_step += 1
                input = Variable(input, requires_grad=False).cuda()
                optimizer.zero_grad()
                optimizer.param_groups[0]["capturable"] = True
                loss = model._loss(input)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
                logging.info("train-epoch %03d %03d %f", epoch, idx, loss)
            logging.info("train-epoch %03d %f", epoch, np.average(losses))
            # utils.save(model, str(weights_dir / f"weights_{epoch}.pt"))
            # utils.save(model, str(weights_dir / f"last.pt"))
            utils.save(model, str(weights_dir / f"{fullname}.pt"))
            
            # if epoch % 50 == 0 and total_step != 0:
            if epoch % 100 and total_step != 0:
                model.eval()
                with torch.no_grad():
                    for idx, (input, img_name) in enumerate(test_loader):
                        input      = Variable(input, volatile=True).cuda()
                        image_name = img_name[0].split("/")[-1].split(".")[0]
                        (
                            L_pred1, L_pred2,
                            L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14,
                            H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H13_H14_diff,
                            H2_blur, H3_blur
                        ) = model(input)
                        input_name = "%s" % image_name
                        H3 = save_images(H3)
                        H2 = save_images(H2)
                        # (debug_dir / "denoise").mkdir(parents=True, exist_ok=True)
                        # (debug_dir / "enhance").mkdir(parents=True, exist_ok=True)
                        # Image.fromarray(H3).save(str(debug_dir / "denoise" / f"{input_name}_denoise_{epoch}.png"), "PNG")
                        # Image.fromarray(H2).save(str(debug_dir / "enhance" / f"{input_name}_enhance_{epoch}.png"), "PNG")
                        (debug_dir / "denoise" / f"{epoch}").mkdir(parents=True, exist_ok=True)
                        (debug_dir / "enhance" / f"{epoch}").mkdir(parents=True, exist_ok=True)
                        Image.fromarray(H3).save(str(debug_dir / "denoise" / f"{epoch}" / f"{input_name}.png"), "PNG")
                        Image.fromarray(H2).save(str(debug_dir / "enhance" / f"{epoch}" / f"{input_name}.png"), "PNG")

# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=_current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
