#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script.
"""

from __future__ import annotations

import argparse
import socket

from one.data import *
from one.nn import ImageEnhancementModel
from one.vision.acquisition import get_image_size
from one.vision.acquisition import ImageLoader
from one.vision.acquisition import to_image
from one.vision.acquisition import write_image_torch
from one.vision.transformation import resize


# H1: - Infer ------------------------------------------------------------------

def preprocess(
    images: Tensor,
    shape : Ints,
) -> tuple[Tensor, Ints, Ints]:
    """
    Preprocessing input.
    
    Args:
        images (Tensor): Input images as [B, H, W, C].
        shape (Ints): Resize images to shape.
        
    Returns:
    	images (Tensor): Input image as  [B, C H, W].
    	size0 (Ints): The original images' sizes.
        size1 (Ints): The resized images' sizes.
    """
    # NOTE: THIS PIPELINE IS FASTER
    size0    = get_image_size(images)
    new_size = to_size(shape)
    if shape:
        images = [resize(image=i, size=new_size) for i in images]
    images = torch.stack(images)
    size1  = get_image_size(images)
    return images, size0, size1


def postprocess_enhancement(
    results: Tensor,
    size0  : Ints,
    size1  : Ints
) -> Tensor:
    """
    Postprocessing results.

    Args:
        results (Tensor): Output images.
        size0 (Ints): The original images' sizes.
        size1 (Ints): The resized images' sizes.
            
    Returns:
        results (Tensor): Post-processed output images as [B, H, W, C].
    """
    if isinstance(results, (list, tuple)):
        results = results[-1]
        
    if size0 != size1:
        results = resize(
            image         = results,
            size          = size0,
            interpolation = InterpolationMode.CUBIC,
        )
    return results


def detect(args: Munch | dict):
    args = Munch.fromDict(args)
    
    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    
    # Input
    data = ImageLoader(source=args.source, batch_size=args.batch_size)
    
    # Output
    output_dir = ""
    if args.save:
        # assert_url_or_file(str(args.root))
        # assert_str(args.name)
        output_dir = Path(args.root) / args.name
    
    # Model
    if is_ckpt_file(args.weights):
        model = MODELS.build(
            name        = args.model,
            cfg         = args.cfg,
            num_classes = args.num_classes,
            phase       = "inference",
        )
        model = model.load_from_checkpoint(
            checkpoint_path = args.weights,
            name            = args.model,
            cfg             = args.cfg,
            num_classes     = args.num_classes,
            phase           = "inference",
        )
    else:
        model = MODELS.build(
            name        = args.model,
            cfg         = args.cfg,
            pretrained  = args.weights,
            num_classes = args.num_classes,
            phase       = "inference",
        )
    devices = select_device(device=args.devices)
    model.to(devices)
    
    print_dict(args, title=model.fullname)
    console.log("[green]Done")
    
    # Visualize
    if args.verbose:
        cv2.namedWindow("image",  cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    
    # H2: - Training -----------------------------------------------------------
    console.rule("[bold red]2. INFERENCE")
    
    with progress_bar() as pbar:
        for batch_idx, batch in pbar.track(
            enumerate(data),
            total=len(data),
            description=f"[bright_yellow] Processing"
        ):
            images, indexes, files, rel_paths = batch
            input, size0, size1 = preprocess(images, shape=args.shape)
            input = input.to(devices)
            pred  = model.forward(input=input)
            
            if isinstance(model, ImageEnhancementModel):
                results = postprocess_enhancement(
                    results = input,
                    size0   = size0,
                    size1   = size1,
                )
                
            if args.verbose:
                images_np  = to_image(image=images,  denormalize=True)
                results_np = to_image(image=results, denormalize=True)
                for i in range(len(images)):
                    cv2.imshow("image",  images_np[i])
                    cv2.imshow("result", results_np[i])
                    cv2.waitKey(1)
            if args.save:
                for i in range(len(images)):
                    write_image_torch(
                        image = results[i],
                        dir   = output_dir,
                        name  = rel_paths[i],
                    )
            
    console.log("[green]Done")
   
    # H2: - Terminate ----------------------------------------------------------
    if args.verbose:
        cv2.destroyAllWindows()


# H1: - Main -------------------------------------------------------------------

hosts = {
	"lp-labdesktop01-ubuntu": {
        "model"      : "zerodce++",
        "cfg"        : "zerodce++.yaml",
        "weights"    : "imagenet",
        "num_classes": None,
        "source"     : DATA_DIR / "lol226",
        "batch_size" : 1,
        "img_size"   : (3, 256, 256),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "lp-labdesktop02-ubuntu": {
        "model"      : "zerodce++",     
        "cfg"        : "zerodce++.yaml",
        "weights"    : "imagenet",
        "num_classes": None,
        "source"     : DATA_DIR / "lol226",
        "batch_size" : 1,
        "img_size"   : (3, 256, 256),
		"devices"    : "0",
        "root"       : RUNS_DIR / "infer",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
    "lp-imac.local": {
        "model"      : "zerodce++",
        "cfg"        : "zerodce++.yaml",
        "weights"    : "imagenet",
        "num_classes": None,
        "source"     : DATA_DIR / "lol226",
        "batch_size" : 1,
        "img_size"   : (3, 256, 256),
		"devices"    : "cpu",
        "root"       : RUNS_DIR / "infer",
        "name"       : "exp",
        "save"       : True,
        "verbose"    : True,
	},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,                             help="Model name")
    parser.add_argument("--cfg",         type=str,                             help="Model config.")
    parser.add_argument("--weights",     type=str,                             help="Weights path.")
    parser.add_argument("--num-classes", type=int,                             help="Number of classes.")
    parser.add_argument("--source",      type=str,                             help="Data source.")
    parser.add_argument("--batch-size",  type=int,                             help="Total Batch size for all GPUs.")
    parser.add_argument("--img-size",    type=int,  nargs="+",                 help="Image sizes.")
    parser.add_argument("--devices",     type=str,                             help="Will be mapped to either gpus, tpu_cores, num_processes or ipus based on the accelerator type.")
    parser.add_argument("--root",        type=str, default=RUNS_DIR / "infer", help="Save results to root/name")
    parser.add_argument("--name",        type=str,                             help="Save results to root/name")
    parser.add_argument("--save",        action="store_true", default=True,    help="Save.")
    parser.add_argument("--verbose",     action="store_true", default=True,    help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    hostname    = socket.gethostname().lower()
    host_args   = Munch(hosts[hostname])
    
    input_args  = vars(parse_args())
    model       = input_args.get("model",       None) or host_args.get("model",       None)
    cfg         = input_args.get("cfg",         None) or host_args.get("cfg",         None)
    weights     = input_args.get("weights",     None) or host_args.get("weights",     None)
    num_classes = input_args.get("num_classes", None) or host_args.get("num_classes", None)
    source      = input_args.get("source",      None) or host_args.get("source",      None)
    batch_size  = input_args.get("batch_size",  None) or host_args.get("batch_size",  None)
    shape       = input_args.get("img_size",    None) or host_args.get("img_size",    None)
    devices     = input_args.get("devices",     None) or host_args.get("devices",     None)
    root        = input_args.get("root",        None) or host_args.get("root",        None)
    name        = input_args.get("name",        None) or host_args.get("name",        None)
    save        = input_args.get("save",        None) or host_args.get("save",        None)
    verbose     = input_args.get("verbose",     None) or host_args.get("verbose",     None)
    
    args = Munch(
        hostname    = hostname,
        model       = model,
        cfg         = cfg,
        weights     = weights,
        num_classes = num_classes,
        source      = source,
        batch_size  = batch_size,
        shape       = shape,
        devices     = devices,
        root        = root,
        name        = name,
        save        = save,
        verbose     = verbose,
    )
    detect(args)
