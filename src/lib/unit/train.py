"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import click
import torch
import torch.backends.cudnn as cudnn

from trainer import MUNIT_Trainer, UNIT_Trainer
from utils import (
    get_all_data_loaders, get_config, prepare_sub_folder, Timer,
    write_2images, write_html, write_loss,
)

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

cudnn.benchmark = True


@click.command()
@click.option("--config",      default='configs/edges2handbags_folder.yaml', type=click.Path(exists=True), help='Path to the config file.')
@click.option("--output-path", default='.', type=click.Path(exists=False), help="outputs path")
@click.option("--resume",      is_flag=True)
@click.option("--trainer",     default='UNIT', type=click.Choice(["MUNIT", "UNIT"], case_sensitive=False), help="MUNIT|UNIT")
def train(
    config     : str,
    output_path: str,
    resume     : bool,
    trainer    : str
):
    # Load experiment setting
    config                   = get_config(config)
    max_iter                 = config["max_iter"]
    display_size             = config["display_size"]
    config["vgg_model_path"] = output_path
    
    # Setup model and data loader
    if trainer == "MUNIT":
        trainer = MUNIT_Trainer(config)
    elif trainer == "UNIT":
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a  = torch.stack([test_loader_a.dataset[i]  for i in range(display_size)]).cuda()
    test_display_images_b  = torch.stack([test_loader_b.dataset[i]  for i in range(display_size)]).cuda()
    
    # Setup logger and output folders
    model_name       = os.path.splitext(os.path.basename(config))[0]
    train_writer     = tensorboardX.SummaryWriter(os.path.join(output_path + "/logs", model_name))
    output_directory = os.path.join(output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(config, os.path.join(output_directory, "config.yaml"))  # copy config file to output folder
    
    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if resume else 0
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
    
            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, config)
                torch.cuda.synchronize()
    
            # Dump training stats in log file
            if (iterations + 1) % config["log_iter"] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)
    
            # Write images
            if (iterations + 1) % config["image_save_iter"] == 0:
                with torch.no_grad():
                    test_image_outputs  = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, "test_%08d" % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, "train_%08d" % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config["image_save_iter"], "images")
    
            if (iterations + 1) % config["image_display_iter"] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, "train_current")
    
            # Save network weights
            if (iterations + 1) % config["snapshot_save_iter"] == 0:
                trainer.save(checkpoint_directory, iterations)
    
            iterations += 1
            if iterations >= max_iter:
                sys.exit("Finish training")
