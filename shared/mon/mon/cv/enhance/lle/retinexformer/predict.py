#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Retinexformer prediction pipeline model for low-light image enhancement.

References:
    - Paper: "Retinexformer: One-stage Retinex-based Transformer for Low-light
      Image Enhancement," ICCV 2023.
    - Code: https://github.com/caiyuanhao1998/Retinexformer
"""

import copy

import box
import cv2
import torch

import mon
from mon import albumentations as A
from mon.core.nn import functional as F
from retinexformer import create_model, parse

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = root_dir / "retinexformer" / "option" / args.cfg
    cfgs     = parse(str(cfg_path), is_train=False)
    
    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    # gpu_list = ",".join(str(x) for x in args.gpus)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    # print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    device = mon.create_device(args.device)
    cfgs["dist"]   = False
    cfgs["device"] = device

    # Seed
    mon.set_random_seed(args.seed)
    
    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = mon.ROOT_DIR / args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
        checkpoint = torch.load(pretrained)
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = create_model(cfgs).net_g
    try:
        model.load_state_dict(checkpoint["params"])
    except:
        new_checkpoint = {}
        for k in checkpoint["params"]:
            new_checkpoint["module." + k] = checkpoint["params"][k]
        model.load_state_dict(new_checkpoint)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)

    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)
    
    # Predict
    factor = 4
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            image  = datapoint["image"]
            h, w   = mon.image.imgsz(image)
            H, W   = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh   = H - h if h % factor != 0 else 0
            padw   = W - w if w % factor != 0 else 0
            image  = F.pad(image, (0, padw, 0, padh), 'reflect')
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            # Unpad images to original dimensions
            enhanced = outputs[:, :, :h, :w]
            enhanced = mon.image.to_array(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.rt.parse_cli_args(root=root_dir)
    data = mon.utils.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.rt.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
