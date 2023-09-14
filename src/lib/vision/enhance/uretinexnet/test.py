#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/AndersonYong/URetinex-Net

from __future__ import annotations

import argparse
import time

import torch.nn as nn
import torchvision.transforms as transforms

import mon
from network.decom import Decom
from network.Math_Module import P, Q
from utils import *

console = mon.console


def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)


class Inference(nn.Module):
    
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # Loading decomposition model
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts.decom_model_low_weights)
        # Loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding(self.opts.unfolding_model_weights)
        # Loading adjustment model
        self.adjust_model    = load_adjustment(self.opts.adjust_model_weights)
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
            # transforms.Resize(1280),
        ]
        self.transform = transforms.Compose(transform)
        # console.log(self.model_Decom_low)
        # console.log(self.model_R)
        # console.log(self.model_L)
        # console.log(self.adjust_model)
        # time.sleep(8)

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):      
            if t == 0:  # Initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else:  # Update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P   = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q   = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def illumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start_time = time.time()
            R, L       = self.unfolding(input_low_img)
            High_L     = self.illumination_adjust(L, self.opts.ratio)
            I_enhance  = High_L * R
            run_time   = (time.time() - start_time)
        return I_enhance, run_time

    def run(self, low_img_path):
        low_img           = self.transform(Image.open(str(low_img_path))).unsqueeze(0)
        enhance, run_time = self.forward(input_low_img=low_img)
        """
        file_name = os.path.basename(self.opts.img_path)
        name      = file_name.split('.')[0]
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output, file_name.replace(name, "%s_%d_URetinexNet"%(name, self.opts.ratio)))
        np_save_TensorImg(enhance, save_path)
        console.log("================================= time for %s: %f============================"%(file_name, p_time))
        """
        return enhance, run_time
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure")
    parser.add_argument("--data",                    type=str, default="./demo/input")
    parser.add_argument("--decom-model-low-weights", type=str, default="./ckpt/init_low.pth")
    parser.add_argument("--unfolding-model-weights", type=str, default="./ckpt/unfolding.pth")
    parser.add_argument("--adjust-model-weights",    type=str, default="./ckpt/L_adjust.pth")
    parser.add_argument("--image-size",              type=int, default=512)
    parser.add_argument("--ratio",                   type=int, default=5)  # ratio are recommended to be 3-5, bigger ratio will lead to over-exposure
    parser.add_argument("--gpu",                     type=int, default=0)
    parser.add_argument("--output-dir",              type=str, default="./demo/output")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    model = Inference(args).cuda()
    
    # Measure efficiency score
    flops, params, avg_time = mon.calculate_efficiency_score(
        model      = model,
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
        image_paths = list(args.data.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                enhanced_image, run_time = model.run(image_path)
                result_path  = args.output_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(result_path))
                sum_time    += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")
