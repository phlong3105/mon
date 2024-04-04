#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import random
import socket

import click
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

import core.logger as Logger
import core.metrics as Metrics
import model as Model
import mon
import options.options as option
from utils import util

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
transform     = transforms.Lambda(lambda t: (t * 2) - 1)


def predict():
    #### options
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to option YMAL file.',
                            default='./config/dataset.yml') # 
    parser.add_argument('--input', type=str, help='testing the unpaired image',
                            default='images/unpaired/')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tfboard', action='store_true')
    parser.add_argument('-c', '--config', type=str, default='config/test_unpaired.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    opt['phase'] = 'test'

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))


    util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    InputPath = args.input
    Image_names = natsort.natsorted(os.listdir(InputPath), alg=natsort.ns.PATH)

    for i in range(len(Image_names)):

        path = InputPath + Image_names[i]
        raw_img = Image.open(path).convert('RGB')
        img_w = raw_img.size[0]
        img_h = raw_img.size[1]
        raw_img = transforms.Resize((img_h // 16 * 16, img_w // 16 * 16))(raw_img)

        raw_img = transform(TF.to_tensor(raw_img)).unsqueeze(0).cuda()

        val_data = {}
        val_data['LQ'] = raw_img
        val_data['GT'] = raw_img
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()
        
        normal_img = Metrics.tensor2img(visuals['HQ']) 
        normal_img = cv2.resize(normal_img, (img_w, img_h))
        ll_img = Metrics.tensor2img(visuals['LQ']) 

        llie_img_mode = 'single'
        if llie_img_mode == 'single':
            # util.save_img(
            #     ll_img, '{}/{}_input.png'.format(result_path, idx))
            util.save_img(
                normal_img, '{}/{}_normal.png'.format(result_path, i+1))
        else:
            util.save_img(
                normal_img, '{}/{}_{}_normal_process.png'.format(result_path, i))
            util.save_img(
                Metrics.tensor2img(visuals['HQ'][-1]), '{}/{}_normal.png'.format(result_path, i))
            normal_img = Metrics.tensor2img(visuals['HQ'][-1])


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    weights  = weights  or args.get("weights")
    project  = args.get("project")
    fullname = fullname or args.get("name")
    device   = device   or args.get("device")
    imgsz    = imgsz    or args.get("imgsz")
    verbose  = verbose  or args.get("verbose")
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    project  = root.name or project
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["data"]       = data
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
    args["device"]     = device
    args["imgsz"]      = imgsz
    args["resize"]     = resize
    args["benchmark"]  = benchmark
    args["save_image"] = save_image
    args["verbose"]    = verbose
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
