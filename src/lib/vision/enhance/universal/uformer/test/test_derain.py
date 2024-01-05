#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "../dataset/"))
sys.path.append(os.path.join(dir_name, ".."))

import utils
from dataset.dataset_motiondeblur import *
from skimage import img_as_ubyte
import cv2

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def net_process(model, image, flip=False):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]

    output = torch.clamp(output, 0, 1).data.cpu().numpy().squeeze().transpose((1, 2, 0))
    return output


def test(args: argparse.Namespace):
    # if args.save_images:
    result_dir_img = os.path.join(args.output_dir, "result")
    utils.mkdir(result_dir_img)

    visualization_dir_img = os.path.join(args.output_dir, "visualization")
    utils.mkdir(visualization_dir_img)

    img_options_val = {"val_h": 1536, "val_w": 2048}
    test_dataset    = get_validation_deblur_data(args.input_dir)
    test_loader     = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    model_restoration = utils.get_arch(args)

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ",     args.weights)

    model_restoration.cuda()
    model_restoration.eval()

    test_patch_size = 128
    stride_rate     = 1 / 2

    with torch.no_grad():
        psnr_val_derain = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            target    = data_test[0].cuda()
            input_    = data_test[1].cuda()
            filenames = data_test[2]

            input_numpy     = np.transpose(input_.squeeze(0).cpu().numpy(), [1, 2, 0])
            ori_h, ori_w, _ = input_numpy.shape

            if ori_h > test_patch_size or ori_w > test_patch_size:
                pad_h      = max(test_patch_size - ori_h, 0)
                pad_w      = max(test_patch_size - ori_w, 0)
                pad_h_half = int(pad_h / 2)
                pad_w_half = int(pad_w / 2)
                if pad_h > 0 or pad_w > 0:
                    input_numpy = cv2.copyMakeBorder(input_numpy, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT)
                new_h, new_w, _ = input_numpy.shape
                stride_h        = int(np.ceil(test_patch_size * stride_rate))
                stride_w        = int(np.ceil(test_patch_size * stride_rate))
                grid_h          = int(np.ceil(float(new_h - test_patch_size) / stride_h) + 1)
                grid_w          = int(np.ceil(float(new_w - test_patch_size) / stride_w) + 1)
                rgb_restored    = np.zeros((new_h, new_w, 3), dtype=float)
                count_crop      = np.zeros((new_h, new_w), dtype=float)
                for index_h in range(0, grid_h):
                    for index_w in range(0, grid_w):
                        s_h        = index_h * stride_h
                        e_h        = min(s_h + test_patch_size, new_h)
                        s_h        = e_h - test_patch_size
                        s_w        = index_w * stride_w
                        e_w        = min(s_w + test_patch_size, new_w)
                        s_w        = e_w - test_patch_size
                        image_crop = input_numpy[s_h:e_h, s_w:e_w].copy()
                        count_crop[s_h:e_h, s_w:e_w] += 1
                        rgb_restored[s_h:e_h, s_w:e_w, :] += net_process(model_restoration, image_crop)
                rgb_restored /= np.expand_dims(count_crop, 2)
                rgb_restored  = rgb_restored[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
                prediction    = cv2.resize(rgb_restored, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_restored  = model_restoration(input_)
                rgb_restored  = torch.clamp(rgb_restored, 0, 1).data.cpu().numpy().squeeze().transpose((1, 2, 0))

            rgb_restored_img = img_as_ubyte(rgb_restored)
            target_img       = img_as_ubyte(target.data.cpu().numpy().squeeze().transpose((1, 2, 0)))
            input_img        = img_as_ubyte(input_.data.cpu().numpy().squeeze().transpose((1, 2, 0)))
            visualization    = np.concatenate([input_img, rgb_restored_img, target_img], axis=0)
            utils.save_img(os.path.join(args.output_dir, "result/", filenames[0] + ".png"), rgb_restored_img)
            utils.save_img(os.path.join(args.output_dir, "visualization/", filenames[0] + ".png"), visualization)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image de-raining evaluation on GT-Rain")
    parser.add_argument("--input-dir",        type=str,            default="fill in your test file with input and groundtruth", help="Directory of validation images")
    parser.add_argument("--output-dir",       type=str,            default="./results/UG2/",     help="Directory for results")
    parser.add_argument("--weights",          type=str,            default="./model_latest.pth", help="Path to weights")  # /mnt/data/yeyuntong/Projects/Transformer/Uformer/logs/motiondeblur/GoPro/Uformer_B_Ours/models/model_latest.pth
    parser.add_argument("--gpus",             type=str,            default="1",                  help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--arch",             type=str,            default="Uformer_B",          help="arch")
    parser.add_argument("--batch-size",       type=int,            default=1,                    help="Batch size for dataloader")
    parser.add_argument("--save-images",      action="store_true",                               help="Save denoised images in result directory")
    parser.add_argument("--embed-dim",        type=int,            default=32,                   help="number of data loading workers")
    parser.add_argument("--win-size",         type=int,            default=8,                    help="number of data loading workers")
    parser.add_argument("--token-projection", type=str,            default="linear",             help="linear/conv token projection")
    parser.add_argument("--token-mlp",        type=str,            default="leff",               help="ffn/leff token mlp")
    parser.add_argument("--dd-in",            type=int,            default=3,                    help="dd_in")
    # args for vit
    parser.add_argument("--vit-dim",          type=int,            default=256,                  help="vit hidden_dim")
    parser.add_argument("--vit-depth",        type=int,            default=12,                   help="vit depth")
    parser.add_argument("--vit-nheads",       type=int,            default=8,                    help="vit hidden_dim")
    parser.add_argument("--vit-mlp-dim",      type=int,            default=512,                  help="vit mlp_dim")
    parser.add_argument("--vit-patch-size",   type=int,            default=16,                   help="vit patch_size")
    parser.add_argument("--global-skip",      action="store_true", default=False,                help="global skip connection")
    parser.add_argument("--local-skip",       action="store_true", default=False,                help="local skip connection")
    parser.add_argument("--vit-share",        action="store_true", default=False,                help="share vit module")
    # args for train
    parser.add_argument("--train_ps",         type=int,            default=128,                  help="patch size of training sample")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test(parse_args())
