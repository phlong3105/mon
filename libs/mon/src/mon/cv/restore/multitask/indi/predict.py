#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements InDi model prediction pipeline for general image restoration.

References:
    - Paper: "Inversion by Direct Iteration: An Alternative to Denoising
      Diffusion for Image Restoration," TMLR 2023.
    - Code: https://github.com/fpramunno/InDI-implementation
"""

import copy

import box

import indi
import mon
import mon.training.albumentations as A

mon.preload()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# --- Predict ---
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Model
    model = indi.InDiUnet(
        in_channels  = 1,
        out_channels = 1,
        time_dim     = 256,
        imgsz        = args.imgsz,
        true_imgsz   = args.imgsz,
    )
    model     = model.to(device)
    diffusion = indi.InDi(imgsz=args.image_size, device=device)
    ema       = indi.EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # Benchmark
    if args.benchmark:
        mon.metrics.benchmark(model)

    # Optimizer
    optimizer = mon.nn.AdamW(model.parameters(), **args.optimizer)

    # Loss
    L = mon.nn.MSELoss()

    # Data I/O
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.build_dataloader(args.data, args.root, transform)

    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with (mon.create_progress_bar() as pbar):
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta  = datapoint["meta"][0]
            path  = mon.Path(meta["path"])
            image = datapoint["image"]
            image = image.to(device)
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            enhanced = mon.image.to_array(enhanced)
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.resolve_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.write(enhanced, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# --- Main ---
def main() -> str:
    cli  = mon.parse_cli_args(root=root_dir)
    data = mon.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
