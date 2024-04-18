#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import mon
from dataloader import *
from lib.vision.enhance.llie.kind.base_parser import BaseParser
from models import *

console = mon.console


def test(args: argparse.Namespace):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.input_dir}")
    
    # Load model
    # args.noDecom = True
    config = mon.load_config_from_file(args.config)
    model  = KinD()
    
    if args.weights is not None:
        if config["noDecom"] is False:
            pretrain_decom = torch.load(args.weights / "kind-decom_net.pth")
            model.decom_net.load_state_dict(pretrain_decom)
            # console.log("Model loaded from decom_net.pth")
        pretrain_restore = torch.load(args.weights / "kind-restore_net.pth")
        model.restore_net.load_state_dict(pretrain_restore)
        # console.log("Model loaded from restore_net.pth")
        pretrain_illum   = torch.load(args.weights / "kind-illum_net.pth")
        model.illum_net.load_state_dict(pretrain_illum)
        # console.log("Model loaded from illum_net.pth")
    
    model = model.cuda()
    model.eval()
    
    # Measure efficiency score
    if args.benchmark:
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
    target_b = 0.70
    with torch.no_grad():
        image_paths = list(args.input_dir.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                image          = Image.open(image_path)
                image          = np.asarray(image, np.float32).transpose((2, 0, 1)) / 255.0
                image          = torch.from_numpy(image).float()
                image          = image.cuda().unsqueeze(0)
                start_time     = time.time()
                bright_low     = torch.mean(image)
                bright_high    = torch.ones_like(bright_low) * target_b + 0.5 * bright_low
                ratio          = torch.div(bright_high, bright_low)
                _, _, enhanced_image = model(L=image, ratio=ratio)
                # enhanced_image = enhanced_image.detach().cpu()[0]
                run_time       = (time.time() - start_time)
                output_path    = args.output_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time      += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")
        
                
if __name__ == "__main__":
    parser = BaseParser()
    args   = parser.parse()
    test(args)
