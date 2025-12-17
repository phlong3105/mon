#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import box
import cv2
import torch

import cyclegan
import mon
import mon.training.albumentations as A

mon.preload()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Hard-code some parameters for test
    opts = cyclegan.TestOptions().parse()  # Get test options
    opts.num_threads    = 0                # Test code only supports num_threads = 0
    opts.batch_size     = 1                # Test code only supports batch_size  = 1
    opts.serial_batches = True             # Disable data shuffling; comment this line if results on randomly chosen images are needed.
    opts.no_flip        = True             # No flip; comment this line if results on flipped images are needed.
    opts.model          = args.network.model
    opts.input_nc       = args.network.input_nc
    opts.output_nc      = args.network.output_nc
    opts.ngf            = args.network.ngf
    opts.ndf            = args.network.ndf
    opts.netD           = args.network.netD
    opts.netG           = args.network.netG
    opts.n_layers_D     = args.network.n_layers_D
    opts.norm           = args.network.norm
    opts.init_type      = args.network.init_type
    opts.init_gain      = args.network.init_gain
    opts.no_dropout     = args.network.no_dropout
    
    # Start
    mon.print_run_summary(args)
    
    # Device
    device      = mon.create_device(args.device)
    opts.device = device
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Pretrained
    pretrained = mon.parse_weights_dir(args.root, args.weights)
    if pretrained and pretrained.is_dir():
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")
        
    # Model
    direction  = args.network.direction
    model_args = {
        "name"   : args.model,
        "opts"   : opts,
        "weights": pretrained,
    }
    model = mon.MODELS.build(**model_args)
    model = model.to(device)
    if opts.eval:
        model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.metrics.benchmark(model)
    
    # Data I/O
    imgsz     = args.imgsz
    transform = A.Compose([
        A.Resize(height=imgsz[0], width=imgsz[1]),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.build_dataloader(args.data, args.root, transform=transform)
    # transform_A = cyclegan.get_transform(opts)
    # transform_B = cyclegan.get_transform(opts)
    
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
            meta_A  = datapoint["meta_A"][0]
            meta_B  = datapoint["meta_B"][0]
            path_A  = mon.Path(meta_A["path"])
            path_B  = mon.Path(meta_B["path"])
            image_A = datapoint["image_A"]
            image_B = datapoint["image_B"]
            image_A = image_A.to(device)
            image_B = image_B.to(device)
            h0, w0  = mon.image.imgsz(meta_A["orig_shape"])
            input_  = {
                "A"      : image_A,
                "B"      : image_B,
                "A_paths": path_A,
                "B_paths": path_B
            }
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            model.set_input(input_)
            model.test()
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            outputs = model.get_current_visuals()
            fake_A  = outputs.get("fake_A")
            fake_B  = outputs.get("fake_B")
            h1, w1  = mon.image.imgsz(fake_A)
            if (h1, w1) != (h0, w0):
                fake_A = cv2.resize(fake_A, (w0, h0))
                fake_B = cv2.resize(fake_B, (w0, h0))
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                # A
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path_A, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / "fake_A" / f"{path_A.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save(fake_A, out_path)
                # B
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path_B, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / "fake_B" / f"{path_A.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save(fake_B, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
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
