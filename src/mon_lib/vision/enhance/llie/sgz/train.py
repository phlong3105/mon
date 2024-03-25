#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import torch.optim

import dataloader
import myloss
from modeling import model
from modeling.fpn import *
from option import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU only
device = get_device()


class Trainer:
    
    def __init__(self):
        self.scale_factor  = args.scale_factor
        self.net           = model.enhance_net_nopool(self.scale_factor, conv_type=args.conv_type).to(device)
        self.seg           = fpn(args.num_of_SegClass).to(device)
        self.seg_criterion = FocalLoss(gamma=2).to(device)
        self.train_dataset = dataloader.lowlight_loader(str(args.input_dir))
        self.train_loader  = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size  = args.train_batch_size,
            shuffle     = True,
            num_workers = args.num_workers,
            pin_memory  = True,
        )
        self.L_color   = myloss.L_color()
        self.L_spa     = myloss.L_spa8(patch_size=args.patch_size)
        self.L_exp     = myloss.L_exp(16)
        self.L_TV      = myloss.L_TV()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.epochs           = args.epochs
        self.E                = args.exp_level
        self.grad_clip_norm   = args.grad_clip_norm
        self.display_iter     = args.display_iter
        self.checkpoints_iter = args.checkpoints_iter
        self.checkpoints_dir  = args.checkpoints_dir

        if args.load_pretrain:
            self.net.load_state_dict(torch.load(args.weights, map_location=device))

    def get_seg_loss(self, enhanced_image):
        # Segment the enhanced image
        seg_input  = enhanced_image.to(device)
        seg_output = self.seg(seg_input).to(device)
        # Build seg output
        target     = (get_NoGT_target(seg_output)).data.to(device)
        # Calculate seg. loss
        seg_loss   = self.seg_criterion(seg_output, target)
        return seg_loss

    def get_loss(self, A, enhanced_image, img_lowlight, E):
        Loss_TV  = 1600 * self.L_TV(A)
        loss_spa = torch.mean(self.L_spa(enhanced_image, img_lowlight))
        loss_col =  5 * torch.mean(self.L_color(enhanced_image))
        loss_exp = 10 * torch.mean(self.L_exp(enhanced_image, E))
        loss_seg = self.get_seg_loss(enhanced_image)
        loss     = Loss_TV + loss_spa + loss_col + loss_exp + 0.1 * loss_seg
        return loss

    def train(self):
        self.net.train()

        for epoch in range(self.epochs):
            for iteration, img_lowlight in enumerate(self.train_loader):
                img_lowlight      = img_lowlight.to(device)
                enhanced_image, A = self.net(img_lowlight)
                loss = self.get_loss(A, enhanced_image, img_lowlight, self.E)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.net.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                if ((iteration + 1) % self.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % self.checkpoints_iter) == 0:
                    torch.save(self.net.state_dict(), args.checkpoints_dir / "best.pt")


if __name__ == "__main__":
    t = Trainer()
    t.train()
