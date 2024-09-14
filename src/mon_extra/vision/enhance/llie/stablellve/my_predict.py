#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/zkawfanx/StableLLVE

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torchvision
from PIL import Image

import mon
from model import UNet

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Model
    model = UNet(n_channels=3, bilinear=True).to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
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
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                image      = Image.open(image_path).convert("RGB")
                image      = (np.asarray(image) / 255.0)
                image      = torch.from_numpy(image).float()
                image      = image.permute(2, 0, 1)
                image      = image.to(device).unsqueeze(0)
                
                # Infer
                timer.tick()
                enhanced_image = model(image)
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(enhanced_image, str(output_path))
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
        
    """
    with torch.no_grad():
        for i, filename in enumerate(filenames):
            test = cv2.imread(filename)/255.0
            test = np.expand_dims(test.transpose([2,0,1]), axis=0)
            test = torch.from_numpy(test).to(device="cuda", dtype=torch.float32)
            out  = model(test)
            out  = out.to(device="cpu").numpy().squeeze()
            out  = np.clip(out*255.0, 0, 255)
            path = filename.replace('/test/','/results/')[:-4]+'.png'
            # folder = os.path.dirname(path)
            # if not os.path.exists(folder):
            #     os.makedirs(folder)
            cv2.imwrite(path, out.astype(np.uint8).transpose([1,2,0]))
            print('%d|%d'%(i+1, len(filenames)))
    """

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    args.weights = args.weights or mon.ZOO_DIR / "vision/enhance/llie/stablellve/stablellve/custom/stablellve_custom_pretrained.pth"
    predict(args)


if __name__ == "__main__":
    main()

# endregion
