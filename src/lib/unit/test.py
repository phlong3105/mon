"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import os
import sys

import click
import torch
import torchvision.utils as vutils
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from trainer import MUNIT_Trainer, UNIT_Trainer
from utils import get_config, pytorch03_to_pytorch04


@click.command()
@click.option("--config",        default=None, type=click.Path(exists=True), help="net configuration.")
@click.option("--input",         default=".", type=click.Path(exists=True), help="input image path.")
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
def test(
    config, input, output_folder, checkpoint, style, a2b, seed, num_style,
    synchronized, output_only, output_path, trainer
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load experiment setting
    config    = get_config(config)
    num_style = 1 if style != '' else num_style
    
    # Setup model and data loader
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
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(checkpoint))
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    
    trainer.cuda()
    trainer.eval()
    encode       = trainer.gen_a.encode if a2b else trainer.gen_b.encode  # encode function
    style_encode = trainer.gen_b.encode if a2b else trainer.gen_a.encode  # encode function
    decode       = trainer.gen_b.decode if a2b else trainer.gen_a.decode  # decode function
    
    if "new_size" in config:
        new_size = config["new_size"]
    else:
        if a2b == 1:
            new_size = config["new_size_a"]
        else:
            new_size = config["new_size_b"]
    
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize(new_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image       = Variable(transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda())
        style_image = Variable(transform(Image.open(style).convert('RGB')).unsqueeze(0).cuda()) if style != '' else None
    
        # Start testing
        content, _ = encode(image)
    
        if trainer == "MUNIT":
            style_rand = Variable(torch.randn(num_style, style_dim, 1, 1).cuda())
            if style != "":
                _, style = style_encode(style_image)
            else:
                style = style_rand
            for j in range(num_style):
                s       = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                path    = os.path.join(output_folder, "output{:03d}.jpg".format(j))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
        elif trainer == "UNIT":
            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            path    = os.path.join(output_folder, "output.jpg")
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        else:
            pass
    
        if not output_only:
            # also save input images
            vutils.save_image(image.data, os.path.join(output_folder, "input.jpg"), padding=0, normalize=True)
