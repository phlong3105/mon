#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements:
    - CycleGAN model for image-to-image translation.
    - Pix2Pix model for image-to-image translation.
    - Colorization model for image colorization (black & white image -> colorful images).

References:
    - Paper: "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017.
    - Paper: "Unpaired Image-to-Image Translation using Cycle-Consistent
      Adversarial Networks," ICCV 2017.
    - Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import copy
import random

import box
import cv2
import torch
from PIL import Image

import mon
import cyclegan

mon.init()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Hard-code some parameters for test
    opts = nerco.TestOptions().parse()  # Get test options
    opts.num_threads    = 0             # Test code only supports num_threads = 0
    opts.batch_size     = 1             # Test code only supports batch_size  = 1
    opts.serial_batches = True          # Disable data shuffling; comment this line if results on randomly chosen images are needed.
    opts.no_flip        = True          # No flip; comment this line if results on flipped images are needed.
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
    mon.rt.print_run_summary(args)
    
    # Device
    device      = mon.create_device(args.device)
    opts.device = device
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Pretrained
    pretrained = mon.rt.parse_weights_dir(args.root, args.weights)
    if pretrained and pretrained.is_dir():
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")
        
    # Model
    # model = create_model(opts)          # create a model given opt.model and other options
    # model = nerco.NeRCo(opts)           # create a model given opt.model and other options
    # model.setup(pretrained, opts)       # regular setup: load and print networks; create schedulers
    model = cyclegan.CycleGAN(opts, pretrained)
    model = model.to(device)
    if opts.eval:
        model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.metrics.benchmark(model)
    
    # Data I/O
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root)
    testB_dir   = root_dir / "nerco" / "dataset" / "testB"
    testB_files = sorted([f for f in testB_dir.glob("*") if f.is_image_file()])
    testB_size  = len(testB_files)
    transform_A = nerco.get_transform(opts)
    transform_B = nerco.get_transform(opts)
    
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
            indexB = random.randint(0, testB_size - 1)
            imageA = Image.open(path).convert("RGB")
            imageB = Image.open(testB_files[indexB]).convert("RGB")
            w0, h0 = imageA.size
            imageA = transform_A(imageA).unsqueeze(0).to(device)
            imageB = transform_B(imageB).unsqueeze(0).to(device)
            dp     = {
                "A"      : imageA,
                "B"      : imageB,
                "A_paths": path,
                "B_paths": testB_files[indexB]
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
            enhanced = outputs.get("fake_B")
            enhanced = nerco.tensor2im(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
            
            '''
            if save_debug:
                if keep_subdirs:
                    rel_path    = image_path.relative_path(data_name)
                    output_path = save_dir / f"{rel_path.parent}_debug"
                else:
                    output_path = save_dir / f"{rel_path.parent}_debug"
                output_path.mkdir(parents=True, exist_ok=True)
                # torchvision.utils.save_image(g_a, str(output_path / f"{image_path.stem}_g_a.jpg"))
                # torchvision.utils.save_image(pre, str(output_path / f"{image_path.stem}_pre.jpg"))
            '''
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
