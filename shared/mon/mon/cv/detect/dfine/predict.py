#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements D-FINE model prediction pipeline for object detection.

References:
    - Paper: "D-FINE: Redefine Regression Task of DETRs as Fine-grained
      Distribution Refinement," ICLR 2025.
    - Code: https://github.com/Peterande/D-FINE
"""

import copy

import box
import torch

import mon
from mon import albumentations as A
import dfine

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Model
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = dfine.DFINE(
        cfg                  = args.cfg,
        weights              = pretrained,
        root                 = args.root,
        device               = device,
        seed                 = args.seed,
        updated_cfg          = args.updated_cfg,
        export_postprocessor = args.export_postprocessor
    )
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)

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
            size0  = torch.tensor([[w0, h0]]).to(device)
            image  = datapoint["image"]
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image, size0)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            labels, boxes, scores = outputs
            labels = [l.cpu().numpy() for l in labels]
            boxes  = [b.cpu().numpy() for b in  boxes]
            scores = [s.cpu().numpy() for s in scores]
            timers.postprocess.tock()

            # Save Result
            if args.save_result:
                out_dir    = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_LABEL_DIR, path, args.keep_subdirs, args.save_nearby)
                label_path = out_dir / f"{path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(label_path), "w") as f:
                    for j, img in enumerate(image):
                        ss = scores[j]
                        cs = labels[j][ss >= args.conf_thres]
                        bs =  boxes[j][ss >= args.conf_thres]
                        if len(bs) == 0:
                            continue
                        bs = mon.hbb.convert(bbox=bs, fmt=mon.BBoxFormat.VOC2YOLO, imgsz=(h0, w0))
                        for c, b, s in zip(cs, bs, ss):
                            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n")
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
