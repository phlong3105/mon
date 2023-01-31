import os
import argparse

import munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

import mon
from .utils import AverageMeter, write_img, chw_to_hwc, pad_img
from .datasets.loader import PairLoader, SingleLoader
from .models import *


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def detect(args: munch.Munch):
    assert args.image is not None and mon.Path(args.image).is_dir()
    image_dir = mon.Path(args.image)
    if args.output is None:
        output_dir = image_dir.parent / "dehaze/gunet"
    else:
        output_dir = mon.Path(args.dst)
    mon.create_dirs(paths=[output_dir])
    
    torch.cuda.empty_cache()
    network = eval(str(args.model))()
    network.cuda()
    network.load_state_dict(single(str(args.weights)))
    network.eval()
    
    predict_dataset = SingleLoader(str(image_dir))
    predict_loader  = DataLoader(
        predict_dataset,
        batch_size      = 1,
        num_workers     = 8,
        pin_memory      = True
    )
    with mon.progress_bar() as pbar:
        for idx, batch in pbar.track(
            sequence    = enumerate(predict_loader),
            total       = len(predict_loader),
            description = f"[bright_yellow] Dehazing"
        ):
            input    = batch["source"].cuda()
            filename = batch["filename"][0]
            with torch.no_grad():
                H, W   = input.shape[2:]
                input  = pad_img(input, network.patch_size if hasattr(network, "patch_size") else 16)
                output = network(input).clamp_(-1, 1)
                output = output[:, :, :H, :W]
                
            out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
            write_img(os.path.join(output_dir, filename), out_img)



# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/train/detection/haze/images",
        help    = "Image directory."
    )
    parser.add_argument(
        "--output",
        type    = str,
        default = None,
        help    = "Output directory."
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = "gunet_b",
        help    = "Model name."
    )
    parser.add_argument(
        "--weights",
        type    = str,
        default = "weights/haze4k/gunet_b.pth",
        help    = "Weights path."
    )
    parser.add_argument(
        "--verbose",
        action = "store_true",
        help   = "Display results."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = munch.Munch.fromDict(vars(parse_args()))
    detect(args=args)
    
# endregion
