#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/AndersonYong/URetinex-Net

from __future__ import annotations

import argparse
import copy
import time

import torchvision.transforms as transforms

import mon
from mon import nn
from network.decom import Decom
from network.Math_Module import P, Q
from utils import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

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
        low_img           = self.transform(Image.open(str(low_img_path)).convert("RGB")).unsqueeze(0)
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
        

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz[0]
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    model = Inference(args).to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    sum_time = 0
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path   = meta["path"]
                enhanced_image, run_time = model.run(image_path)
                output_path  = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time    += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    args.decom_model_low_weights = mon.ZOO_DIR / "vision/enhance/llie/uretinexnet/uretinexnet/lol_v1/uretinexnet_lol_v1_init_low.pth"
    args.unfolding_model_weights = mon.ZOO_DIR / "vision/enhance/llie/uretinexnet/uretinexnet/lol_v1/uretinexnet_lol_v1_unfolding.pth"
    args.adjust_model_weights    = mon.ZOO_DIR / "vision/enhance/llie/uretinexnet/uretinexnet/lol_v1/uretinexnet_lol_v1_l_adjust.pth"
    args.ratio = 5
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
