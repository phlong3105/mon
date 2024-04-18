#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import random
import socket
import time
from os import path as osp

import click
import numpy as np
import torch

import mon
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (
    check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs,
    MessageLogger, mkdir_and_rename, set_random_seed,
)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def init_loggers(opt):
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger   = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (
        (opt["logger"].get("wandb") is not None) and 
        (opt["logger"]["wandb"].get("project") is not None) and 
        ("debug" not in opt["name"])
    ):
        assert opt["logger"].get("use_tb_logger") is True, "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        tb_logger = init_tb_logger(log_dir=osp.join(opt["path"]["tb_logger"], opt["name"]))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # Create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            train_set     = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set, 
                opt["world_size"],
                opt["rank"],
                dataset_enlarge_ratio,
            )
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu = opt["num_gpu"],
                dist    = opt["dist"],
                sampler = train_sampler,
                seed    = opt["manual_seed"],
            )
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt["batch_size_per_gpu"] * opt["world_size"])
            )
            total_iters  = int(opt["train"]["total_iter"])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                "Training statistics:"
                f"\n\tNumber of train images: {len(train_set)}"
                f"\n\tDataset enlarge ratio: {dataset_enlarge_ratio}"
                f"\n\tBatch size per gpu: {dataset_opt['batch_size_per_gpu']}"
                f"\n\tWorld size (gpu number): {opt['world_size']}"
                f"\n\tRequire iter number per epoch: {num_iter_per_epoch}"
                f"\n\tTotal epochs: {total_epochs}; iters: {total_iters}."
            )
        elif phase == "val":
            val_set    = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu = opt["num_gpu"],
                dist    = opt["dist"],
                sampler = None,
                seed    = opt["manual_seed"]
            )
            logger.info(f"Number of val images/folders in {dataset_opt['name']}: {len(val_set)}")
        else:
            # raise ValueError(f"Dataset phase {phase} is not recognized.")
            pass

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def train(args: argparse.Namespace):
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    device   = args.device
    launcher = args.launcher
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Override options with args
    opt = parse(args.opt, is_train=True)
    
    experiments_root = save_dir
    opt["path"]["root"]               = str(experiments_root)
    opt["path"]["experiments_root"]   = str(experiments_root)
    opt["path"]["models"]             = str(experiments_root / "weights")
    opt["path"]["training_states"]    = str(experiments_root / "training_state")
    opt["path"]["log"]                = str(experiments_root)
    opt["path"]["tb_logger"]          = str(experiments_root / "tb_logger")
    opt["logger"]["wandb"]            = str(experiments_root / "wandb")
    opt["path"]["visualization"]      = str(experiments_root / "visualization")
    opt["path"]["pretrain_network_g"] = str(weights)
    
    opt["datasets"]["train"]["dataroot_gt"] = mon.DATA_DIR / opt["datasets"]["train"]["dataroot_gt"]
    opt["datasets"]["train"]["dataroot_lq"] = mon.DATA_DIR / opt["datasets"]["train"]["dataroot_lq"]
    opt["datasets"]["val"]["dataroot_gt"]   = mon.DATA_DIR / opt["datasets"]["val"]["dataroot_gt"]
    opt["datasets"]["val"]["dataroot_lq"]   = mon.DATA_DIR / opt["datasets"]["val"]["dataroot_lq"]

    opt["device"] = device
    
    if launcher == "none":
        opt["dist"] = False
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if launcher == "slurm" and "dist_params" in opt:
            init_dist(launcher, **opt["dist_params"])
        else:
            init_dist(launcher)
            print("init dist .. ", launcher)
    
    opt["rank"], opt["world_size"] = get_dist_info()
    
    # Random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])
    
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    # Automatic resume ..
    state_folder_path = opt["path"]["training_states"]
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = "{}.state".format(max([int(x[0:-6]) for x in states]))
        resume_state   = os.path.join(state_folder_path, max_state_file)
        opt["path"]["resume_state"] = resume_state

    # Load resume states if necessary
    if opt["path"].get("resume_state"):
        device_id    = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
    else:
        resume_state = None
    
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"] and opt["rank"] == 0:
            mkdir_and_rename(osp.join(opt["path"]["tb_logger"], opt["name"]))
    
    # Initialize loggers
    logger, tb_logger = init_loggers(opt)

    # Create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # Create model
    if resume_state:  # resume training
        check_resume(opt, resume_state["iter"])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']}, "
            f"iter: {resume_state['iter']}."
        )
        start_epoch  = resume_state["epoch"]
        current_iter = resume_state["iter"]
    else:
        model        = create_model(opt)
        start_epoch  = 0
        current_iter = 0

    # Create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    
    # Dataloader prefetcher
    prefetch_mode  = opt["datasets"]["train"].get("prefetch_mode")
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f"Use {prefetch_mode} prefetch dataloader")
        if opt["datasets"]["train"].get("pin_memory") is not True:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}."
            f"Supported ones are: None, 'cuda', 'cpu'."
        )

    # Training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_time  = time.time()
    iter_time  = time.time()
    start_time = time.time()

    # For epoch in range(start_epoch, total_epochs + 1):
    iters            = opt["datasets"]["train"].get("iters")
    batch_size       = opt["datasets"]["train"].get("batch_size_per_gpu")
    mini_batch_sizes = opt["datasets"]["train"].get("mini_batch_sizes")
    gt_size          = opt["datasets"]["train"].get("gt_size")
    mini_gt_sizes    = opt["datasets"]["train"].get("gt_sizes")

    groups   = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])
    logger_j = [True] * len(groups)
    scale    = opt["scale"]
    epoch    = start_epoch
    
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time     = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt["train"].get("warmup_iter", -1))

            # Progressive learning ---------------------
            j = ((current_iter>groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]
            
            mini_gt_size    = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            
            if logger_j[bs_j]:
                logger.info("\n Updating Patch_Size to {} and Batch_Size to {} \n".format(mini_gt_size, mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            lq = train_data["lq"]
            gt = train_data["gt"]

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            # -------------------------------------------

            model.feed_train_data({"lq": lq, "gt": gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs" : model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if opt.get("val") is not None and (current_iter % opt["val"]["val_freq"] == 0):
                rgb2bgr   = opt["val"].get("rgb2bgr",   True)
                # wheather use uint8 image to compute metrics
                use_image = opt["val"].get("use_image", True)
                model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"], rgb2bgr, use_image)

            data_time  = time.time()
            iter_time  = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch
    
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get("val") is not None:
        model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"])
    if tb_logger:
        tb_logger.close()

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/train/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--local-rank", type=int, default=0)
@click.option("--launcher",   type=click.Choice(["none", "pytorch", "slurm"]), default="none", help="Job launcher.")
@click.option("--epochs",     type=int, default=300,  help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok",   is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    local_rank: int,
    launcher  : str,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    project  = root.name
    save_dir = save_dir or root / "run" / "train" / fullname
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    
    # Update arguments
    args = {
        "root"      : root,
        "config"    : config,
        "opt"       : config,
        "weights"   : weights,
        "model"     : model,
        "project"   : project,
        "fullname"  : fullname,
        "save_dir"  : save_dir,
        "device"    : device,
        "local_rank": local_rank,
        "launcher"  : launcher,
        "epochs"    : epochs,
        "steps"     : steps,
        "exist_ok"  : exist_ok,
        "verbose"   : verbose,
    }
    args = argparse.Namespace(**args)
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(args.save_dir))
    
    train(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
