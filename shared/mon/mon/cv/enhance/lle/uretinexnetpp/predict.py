#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements URetinex-Net++ model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Interpretable Optimization-Inspired Unfolding Network for Low-light
      Image Enhancement," IEEE TPAMI 2025.
    - Code: https://github.com/AndersonYong/URetinex-Net-PLUS
"""

import copy

import box
import cv2
import torch

import mon
import uretinexnetpp

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)
 
 
# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    args.Decom_model_high_path         = mon.ZOO_DIR / args.decom_model_high_weights
    args.Decom_model_low_path          = mon.ZOO_DIR / args.decom_model_low_weights
    args.fusion_model_A_path           = mon.ZOO_DIR / args.fusion_weights
    args.pretrain_unfolding_model_path = mon.ZOO_DIR / args.pretrain_unfolding_weights
    '''
    state_dict = torch.load(str(args["fusion_model_A_path"]), weights_only=False)
    print(state_dict.keys())
    print(state_dict["opts"])
    state_dict["opts"].Decom_model_low_path = args["Decom_model_low_path"]
    torch.save(state_dict, str(args["fusion_model_A_path"]))
    
    state_dict = torch.load(str(args["pretrain_unfolding_model_path"]), weights_only=False)
    print(state_dict.keys())
    print(state_dict["opts"])
    state_dict["opts"].Decom_model_low_path = args["Decom_model_low_path"]
    torch.save(state_dict, str(args["pretrain_unfolding_model_path"]))
    '''

    # Model
    model = uretinexnetpp.URetinexNetPP(args)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    data_name, dataset = mon.data.build_dataset(args.data, args.root)

    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataset),
            total       = len(dataset),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model.run(path)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs[0]
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
