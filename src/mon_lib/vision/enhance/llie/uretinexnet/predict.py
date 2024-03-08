#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/AndersonYong/URetinex-Net

from __future__ import annotations

import argparse
import socket
import time

import click
import torchvision.transforms as transforms

from mon import core, data as d, nn
from mon.globals import ZOO_DIR
from network.decom import Decom
from network.Math_Module import P, Q
from utils import *

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


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
    data      = args.data
    save_dir  = args.save_dir
    device    = args.device
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Inference(args).to(device)
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = nn.calculate_efficiency_score(
            model      = model,
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
    console.log(f"{data}")
    data_name, data_loader, data_writer = d.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    with torch.no_grad():
        sum_time = 0
        with core.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path   = meta["image_path"]
                enhanced_image, run_time = model.run(image_path)
                output_path  = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time    += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Prioritize input args --> config file args
    root     = core.Path(root)
    decom_model_low_weights = ZOO_DIR / "vision/enhance/llie/uretinexnet/uretinexnet_init_low.pth"
    unfolding_model_weights = ZOO_DIR / "vision/enhance/llie/uretinexnet/uretinexnet_unfolding.pth"
    adjust_model_weights    = ZOO_DIR / "vision/enhance/llie/uretinexnet/uretinexnet_L_adjust.pth"
    project  = root.name
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = core.Path(save_dir)
    device   = core.parse_device(device)
    ratio    = 5
    # imgsz    = core.str_to_int_list(imgsz)
    # imgsz    = [int(i) for i in imgsz]
    imgsz    = core.parse_hw(imgsz)[0]
    
    # Update arguments
    args = {
        "root"                   : root,
        "config"                 : config,
        "decom_model_low_weights": decom_model_low_weights,
        "unfolding_model_weights": unfolding_model_weights,
        "adjust_model_weights"   : adjust_model_weights,
        "model"                  : model,
        "data"                   : data,
        "project"                : project,
        "name"                   : fullname,
        "save_dir"               : save_dir,
        "ratio"                  : ratio,
        "device"                 : device,
        "imgsz"                  : imgsz,
        "resize"                 : resize,
        "benchmark"              : benchmark,
        "save_image"             : save_image,
        "verbose"                : verbose
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
