#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements UEC model prediction pipeline for unsupervised exposure correction.

References:
    - Paper: "Unsupervised Exposure Correction," ECCV 2024.
    - Code: https://github.com/BeyondHeaven/uec_code
"""

import copy

import box
import cv2
import torch

import mon
import uec
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Hard-code some parameters for test
    cfgs                = uec.TestOptions().parse()  # get test options
    cfgs.num_threads    = 0             # test code only supports num_threads = 0
    cfgs.batch_size     = 1             # test code only supports batch_size  = 1
    cfgs.serial_batches = True          # disable data shuffling; comment this line if results on randomly chosen images are needed.
    cfgs.no_flip        = True          # no flip; comment this line if results on flipped images are needed.
    cfgs.display_id     = -1            # no visdom display; the test code saves the results to a HTML file.
    
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device      = mon.create_device(args.device)
    cfgs.device = device
    
    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = mon.rt.parse_weights_dir(args.root, args.weights)
    if pretrained and pretrained.is_dir():
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")
    
    # Model
    model = uec.UEC(cfgs, pretrained)
    model = model.to(device)
    if cfgs.eval:
        model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    # imgsz     = args.imgsz if args.resize else (0, 0)
    imgsz     = 256 if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)
    
    ref_image = root_dir / "uec" / "dataset" / "testB" / "a0001-jmac_DSC1459.jpg"
    ref_image = mon.image.load_image(ref_image)
    ref_image = transform(image=ref_image)["image"]
    ref_image = ref_image.unsqueeze(0).to(device)
    
    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            image  = datapoint["image"]
            image  = image.to(device)
            dp     = {
                "image_pair": [image, ref_image],
                "image_path": path
            }
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            model.set_input(dp)
            model.test()
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            outputs  = model.get_current_visuals()
            enhanced = outputs.get("fake_img")
            enhanced = mon.image.to_array(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            if args.save_debug:
                image = mon.image.to_array(image)
                ref   = datapoint.get("ref", None)
                ref   = mon.image.to_array(ref) if ref is not None else None
                if (h1, w1) != (h0, w0):
                    image = cv2.resize(image, (w0, h0))
                    ref   = cv2.resize(ref,   (w0, h0)) if ref is not None else None
                if ref is not None:
                    debug_image = cv2.hconcat([image, enhanced, ref])
                else:
                    debug_image = cv2.hconcat([image, enhanced])
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
            # Save Debug
            if args.save_debug:
                debug_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                debug_path = debug_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                mon.image.save_image(debug_image, debug_path)
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
