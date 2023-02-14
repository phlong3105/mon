"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import click
import torchvision.utils as vutils
from torch.autograd import Variable

from data import ImageFolder
from trainer import MUNIT_Trainer, UNIT_Trainer
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import sys
import torch
import os


@click.command()
@click.option("--config",        default="configs/edges2handbags_folder", type=click.Path(exists=True), help="Path to the config file.")
@click.option("--input-folder",  default=".", type=click.Path(exists=True), help="input image folder.")
@click.option("--output-folder", default=".", type=click.Path(exists=False), help="output image path.")
@click.option("--checkpoint",    default=None, type=click.Path(exists=False), help="checkpoint of autoencoders")
@click.option("--style",         default="", type=str, help="style image path")
@click.option("--a2b",           default=1, type=int, help="1 for a2b and others for b2a")
@click.option("--seed",          default=10, type=int, help="random seed")
@click.option("--num_style",     default=10, type=int, help="number of styles to sample")
@click.option("--synchronized",  is_flag=True, help="whether use synchronized style code or not")
@click.option("--output_only",   is_flag=True, help="whether use synchronized style code or not")
@click.option("--output_path",   default=".", type=click.Path(exists=False), help="path for logs, checkpoints, and VGG model weight")
@click.option("--trainer",       default="UNIT", type=click.Choice(["MUNIT", "UNIT"], case_sensitive=False), help="MUNIT|UNIT")
def test_batch(
    config, input_folder, output_folder, checkpoint, style, a2b, seed,
    num_style, synchronized, output_only, output_path, trainer
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Load experiment setting
    config    = get_config(config)
    input_dim = config["input_dim_a"] if a2b else config["input_dim_b"]
    
    # Setup model and data loader
    image_names = ImageFolder(input_folder, transform=None, return_paths=True)
    data_loader = get_data_loader_folder(input_folder, 1, False, new_size=config["new_size_a"], crop=False)
    
    config["vgg_model_path"] = output_path
    if trainer == "MUNIT":
        style_dim = config["gen"]["style_dim"]
        trainer = MUNIT_Trainer(config)
    elif trainer == "UNIT":
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
    
    try:
        state_dict = torch.load(checkpoint)
        trainer.gen_a.load_state_dict(state_dict["a"])
        trainer.gen_b.load_state_dict(state_dict["b"])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(checkpoint))
        trainer.gen_a.load_state_dict(state_dict["a"])
        trainer.gen_b.load_state_dict(state_dict["b"])
    
    trainer.cuda()
    trainer.eval()
    encode = trainer.gen_a.encode if a2b else trainer.gen_b.encode  # encode function
    decode = trainer.gen_b.decode if a2b else trainer.gen_a.decode  # decode function
    
    if trainer == "MUNIT":
        # Start testing
        style_fixed = Variable(torch.randn(num_style, style_dim, 1, 1).cuda(), volatile=True)
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            print(names[1])
            images = Variable(images.cuda(), volatile=True)
            content, _ = encode(images)
            style = style_fixed if synchronized else Variable(torch.randn(num_style, style_dim, 1, 1).cuda(), volatile=True)
            for j in range(num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                # path = os.path.join(output_folder, "input{:03d}_output{:03d}.jpg".format(i, j))
                basename = os.path.basename(names[1])
                path     = os.path.join(output_folder+"_%02d"%j,basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not output_only:
                # also save input images
                vutils.save_image(images.data, os.path.join(output_folder, "input{:03d}.jpg".format(i)), padding=0, normalize=True)
    elif trainer == "UNIT":
        # Start testing
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            print(names[1])
            images     = Variable(images.cuda(), volatile=True)
            content, _ = encode(images)
    
            outputs  = decode(content)
            outputs  = (outputs + 1) / 2.0
            # path = os.path.join(output_folder, "input{:03d}_output{:03d}.jpg".format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(output_folder,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not output_only:
                # also save input images
                vutils.save_image(images.data, os.path.join(output_folder, "input{:03d}.jpg".format(i)), padding=0, normalize=True)
    else:
        pass
