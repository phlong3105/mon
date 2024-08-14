#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time
from os import listdir
from os.path import join, basename

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as Data
from PIL import Image
from torch.autograd import Variable
from torchvision import utils as vutils

import load_data as DA
from lib.vision.enhance.les.jin2022.utils import is_image_file
from net import *
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import utils as vutils

import load_data as DA
from mon import RUN_DIR
from net import *

console = mon.console


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class ExclusionLoss(nn.Module):
    def __init__(self, level=3):
        super(ExclusionLoss, self).__init__()
        self.level    = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid  = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            alphay   = 1
            alphax   = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


def gradient(pred):
    D_dy      = pred[:, :, 1:]    - pred[:, :, :-1]
    D_dx      = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


def smooth_loss(pred_map):
    dx,   dy   = gradient(pred_map)
    dx2,  dxdy = gradient(dx)
    dydx, dy2  = gradient(dy)
    loss       = (dx2.abs().mean()  + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())
    return loss


def rgb2gray(rgb):
    gray = 0.2989 * rgb[:, :, 0:1, :] + \
           0.5870 * rgb[:, :, 1:2, :] + \
           0.1140 * rgb[:, :, 2:3, :]
    return gray


def demo(args, dle_net, optimizer_dle_net, inputs):
    dle_net.train()
    img_in = Variable(torch.FloatTensor(inputs["img_in"])).cuda()
    optimizer_dle_net.zero_grad()

    le_pred  = dle_net(img_in)
    dle_pred = img_in + le_pred

    lambda_cc      = 1.0
    dle_pred_cc    = torch.mean(dle_pred, dim=1, keepdims=True)
    cc_loss        = (F.l1_loss(dle_pred[:, 0:1, :, :], dle_pred_cc) + \
                      F.l1_loss(dle_pred[:, 1:2, :, :], dle_pred_cc) + \
                      F.l1_loss(dle_pred[:, 2:3, :, :], dle_pred_cc))*(1/3) ##Color Constancy Loss

    lambda_recon   = 1.0
    recon_loss     = F.l1_loss(dle_pred, img_in)

    lambda_excl    = 0.01
    data_type      = torch.cuda.FloatTensor
    excl_loss      = ExclusionLoss().type(data_type)

    lambda_smooth  = 1.0
    le_smooth_loss = smooth_loss(le_pred)

    loss  = lambda_recon * recon_loss + lambda_cc * cc_loss
    loss += lambda_excl * excl_loss(dle_pred, le_pred)
    loss += lambda_smooth * le_smooth_loss
    loss.backward()

    optimizer_dle_net.step()

    imgs_dict = {
        "dle_pred": dle_pred.detach().cpu(),
    }
    return imgs_dict


def main(args: argparse.Namespace):
    args.use_gray       = False
    args.input_dir      = mon.Path(args.input_dir)
    args.output_dir     = mon.Path(args.output_dir)
    args.checkpoint_dir = mon.Path(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    # Initialize model
    channels = 1 if args.use_gray else 3
    dle_net  = Net(input_nc=channels, output_nc=channels)

    # Measure efficiency score
    if args.benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = dle_net,
            image_size = args.image_size,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")

    # Load weights
    dle_net = Net(input_nc=channels, output_nc=channels)
    dle_net = nn.DataParallel(dle_net).cuda()
    if args.weights is not None:
        dle_net.load_state_dict(torch.load(str(args.weights))["state_dict"])

    optimizer_dle_net = optim.Adam(dle_net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    #
    num_items    = 0
    sum_time     = 0
    in_filenames = sorted([join(args.input_dir, x) for x in listdir(args.input_dir) if is_image_file(x)])
    with mon.get_progress_bar() as pbar:
        for in_filename in pbar.track(
            sequence    = in_filenames,
            total       = len(in_filenames),
            description = f"[bright_yellow] Inferring"
        ):
            img_name   = basename(in_filename)
            da_list    = sorted([(args.input_dir / file) for file in os.listdir(args.input_dir) if file == img_name])
            da_list    = [str(path) for path in da_list]
            num_items += len(da_list)
            demo_list  = da_list
            demo_list  = demo_list * args.iters
            loader     = torch.utils.data.DataLoader(
                DA.LoadImgs(args, demo_list, mode="demo"),
                batch_size  = 1,
                shuffle     = True,
                num_workers = 16,
                drop_last   = False,
            )

            count_idx  = 0
            start_time = time.time()
            with mon.get_progress_bar() as pbar:
                for batch_idx, (inputs, img_in_path) in loader:
                    count_idx = count_idx + 1
                    imgs_dict = demo(args, dle_net, optimizer_dle_net, inputs)

                    if count_idx % args.iters == 0:
                        img_in_path = mon.Path(img_in_path[0])
                        inout       = args.output_dir / f"{img_in_path.stem}_in_out.png"
                        out         = args.output_dir / f"{img_in_path.stem}_out.png"
                        save_img    = torch.cat((inputs["img_in"][0, :, :, :], imgs_dict["dle_pred"][0, :, :, :]), dim=2)
                        in_img      = inputs["img_in"][0, :, :, :]
                        out_img     = imgs_dict["dle_pred"][0, :, :, :]
                        vutils.save_image(save_img, str(inout))
                        vutils.save_image(out_img,  str(out))
                        torch.save(dle_net.state_dict(), str(args.checkpoint_dir / "best.pt"))
            run_time = (time.time() - start_time)
            sum_time += run_time
    avg_time = float(sum_time / num_items)
    console.log(f"Average time: {avg_time}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",      type=str,   default="./light-effects",                              help="Image to be used for demo")
    parser.add_argument("--output-dir",     type=str,   default=RUN_DIR / "predict/vision/enhance/les/jin2022", help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--weights",        type=str,   default=None,                                           help="model to initialize with")
    parser.add_argument("--image-size",     type=int,   default=512,                                            help="The training size of image")
    parser.add_argument("--load-size",      type=str,   default="Resize",                                       help="Width and height to resize training and testing frames. None for no resizing, only [512, 512] for no resizing")
    parser.add_argument("--crop-size",      type=str,   default="[512, 512]",                                   help="Width and height to crop training and testing frames. Must be a multiple of 16")
    parser.add_argument("--iters",          type=int,   default=60,                                             help="No of iterations to train the model.")
    parser.add_argument("--lr",             type=float, default=1e-4,                                           help="Learning rate for the model.")
    parser.add_argument("--checkpoint-dir", type=str,   default=RUN_DIR / "train/vision/enhance/les/jin2022",   help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--benchmark",      action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
