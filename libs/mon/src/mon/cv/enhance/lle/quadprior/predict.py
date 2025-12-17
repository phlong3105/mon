#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements QuadPrior model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
      Priors," CVPR 2024.
    - Code: https://github.com/daooshee/QuadPrior
"""

import copy
import random

import box
import cv2
import einops
import numpy as np
import torch
import torch.utils
from pytorch_lightning import seed_everything

import mon
from mon import nn
from quadprior import (
    create_model, DPMSolverSampler, HWC3, load_state_dict, resize_image,
)

mon.preload()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def benchmark(model: nn.Module):
    params, macs, flops = mon.metrics.compute_model_stats(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"MACs      : {macs:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


def process(
    model,
    diffusion_sampler,
    input_image,
    prompt          : str   = "",
    num_samples     : int   = 1,
    image_resolution: int   = 512,
    diffusion_steps : int   = 10,
    guess_mode      : bool  = False,
    strength        : float = 1.0,
    scale           : float = 9.0,
    seed            : int   = 0,
    eta             : float = 0.0,
    use_float16     : bool  = True,
):
    with torch.no_grad():
        detected_map = resize_image(HWC3(input_image), image_resolution)
        H, W, C      = detected_map.shape
        
        if use_float16:
            control = torch.from_numpy(detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
        else:
            control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        ae_hs   = model.encode_first_stage(control * 2 - 1)[1]
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        # if args.save_memory:
        #     model.low_vram_shift(is_diffusing=False)
        
        cond    = {
            "c_concat"   : [control],
            "c_crossattn": [model.get_unconditional_conditioning(num_samples)]
        }
        un_cond = {
            "c_concat"   : None if guess_mode else [control],
            "c_crossattn": [model.get_unconditional_conditioning(num_samples)]
        }
        shape   = (4, H // 8, W // 8)
        
        # if args.save_memory:
        #     model.low_vram_shift(is_diffusing=True)
        
        model.control_scales   = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = diffusion_sampler.sample(
            diffusion_steps, num_samples, shape, cond,
            verbose                      = False,
            eta                          = eta,
            unconditional_guidance_scale = scale,
            unconditional_conditioning   = un_cond,
            dmp_order                    = 3,
        )
        
        # if args.save_memory:
        #     model.low_vram_shift(is_diffusing=False)
        
        if use_float16:
            x_samples = model.decode_new_first_stage(samples.to(dtype=torch.float16), ae_hs)
        else:
            x_samples = model.decode_new_first_stage(samples, ae_hs)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [x_samples[i] for i in range(num_samples)]
    return results


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    cfg_path  = root_dir / "src" / "models" / args.cfg
    init_ckpt = mon.ROOT_DIR / "zoo/cv/enhance/lle/quadprior/quadprior/coco80/control_sd15_init.ckpt"
    ae_ckpt   = mon.ROOT_DIR / "zoo/cv/enhance/lle/quadprior/quadprior/coco80/ae_epoch=00_step=7000.ckpt"

    model          = create_model(config_path=cfg_path).cpu()
    state_dict     = load_state_dict(str(init_ckpt), location="cpu")
    new_state_dict = {}
    for s in state_dict:
        if "cond_stage_model.transformer" not in s:
            new_state_dict[s] = state_dict[s]
    model.load_state_dict(new_state_dict)
    model.add_new_layers()  # Insert new layers in ControlNet (sorry for the ugliness)

    # Load trained checkpoint
    state_dict     = load_state_dict(pretrained, location="cpu")
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if "_forward_module.control_model" in sd_name:
            new_state_dict[sd_name.replace("_forward_module.control_model.", "")] = sd_param
    model.control_model.load_state_dict(new_state_dict)
    model.change_first_stage(ae_ckpt)  # Load bypass decoder
    
    if args.use_float16:
        model = model.to(device).to(dtype=torch.float16)
    else:
        model = model.to(device)
    diffusion_sampler = DPMSolverSampler(model)
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Data I/O
    data_name, dataloader = mon.build_dataloader(args.data, args.root)

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
            image  = datapoint["image"][0]
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            # If you set num_samples > 1, process will return multiple results
            outputs = process(
                model, diffusion_sampler,
                input_image      = image,
                num_samples      = 1,
                image_resolution = args.imgsz[0],
                use_float16      = args.use_float16,
            )[0]
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            h1, w1  = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save(enhanced, out_path)
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
