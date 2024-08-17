#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import torch
import torch.optim
import argparse
import glob
import logging
import os
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import utils
from dataset import ImageLowSemDataset, ImageLowSemDataset_Val
from model import *
import dataloader
import model as mmodel
import mon
import myloss

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Module

class GradCAM:
    
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, sem, depth, target_output):
        self.model.zero_grad()
        output = self.model(input_image, sem, depth)
        
        target = target_output  # 使用目标输出计算梯度
        target.backward()
        
        gradients   = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights     = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def visualize_cam_on_image(image, cam, save_path):
    image        = image.cpu().numpy().transpose(1, 2, 0)
    cam          = cv2.resize(cam, (image.shape[1], image.shape[0]))  # 确保 cam 尺寸与 image 一致
    heatmap      = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap      = np.float32(heatmap) / 255
    cam_on_image = heatmap + np.float32(image)
    cam_on_image = cam_on_image / np.max(cam_on_image)
    
    plt.imshow(np.uint8(255 * cam_on_image))
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

# endregion


# region Train

def train(args: argparse.Namespace):
    # General config
    save_dir   = mon.Path(args.save_dir)
    weights    = args.weights
    device     = mon.set_device(args.device)
    epochs     = args.epochs
    verbose    = args.verbose
    seed       = args.seed
    lr         = args.lr
    stage      = args.stage
    pretrained = args.pretrained
    arch_      = args.arch_
    frozen     = args.frozen
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    DCE_net = mmodel.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    if weights is not None and mon.Path(weights).is_weights_file():
        DCE_net.load_state_dict(torch.load(weights))
    DCE_net.train()
    
    # Loss
    L_color = myloss.L_color()
    L_spa   = myloss.L_spa()
    L_exp   = myloss.L_exp(16 , 0.6)
    L_tv    = myloss.L_TV()
    
    # Optimizer
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Data I/O
    train_dataset = dataloader.lowlight_loader(args.data)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.train_batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True
    )
    
    # Training
    with mon.get_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Predicting"
        ):
            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.to(device)
                enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
                
                loss_tv  = 200 * L_tv(A)
                loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col = 5   * torch.mean(L_color(enhanced_image))
                loss_exp = 10  * torch.mean(L_exp(enhanced_image))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), args.grad_clip_norm)
                optimizer.step()
                
                if ((iteration + 1) % args.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % args.checkpoints_iter) == 0:
                    torch.save(DCE_net.state_dict(), weights_dir / "best.pt")

# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
