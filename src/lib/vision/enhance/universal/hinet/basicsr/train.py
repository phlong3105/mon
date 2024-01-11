#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime
import logging
import math
import random
import time
from os import path as osp

import torch

import mon
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (
    MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
    init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename,
    set_random_seed
)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

console = mon.console


def init_loggers(args):
    log_file = osp.join(args["path"]["log"], f"train_{args['name']}_{get_time_str()}.log")
    logger   = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(args))

    # Initialize wandb logger before tensorboard logger to allow proper sync:
    if (args["logger"].get("wandb") is not None) and (args["logger"]["wandb"].get("project") is not None) and ("debug" not in args["name"]):
        assert args["logger"].get("use_tb_logger") is True, "Should turn on tensorboard when using wandb"
        init_wandb_logger(args)
    tb_logger = None
    if args["logger"].get("use_tb_logger") and "debug" not in args["name"]:
        tb_logger = init_tb_logger(log_dir=osp.join("tb_logger", args["name"]))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            train_set     = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio)
            train_loader  = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu = opt["num_gpu"],
                dist    = opt["dist"],
                sampler = train_sampler,
                seed    = opt["manual_seed"]
            )

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt["batch_size_per_gpu"] * opt["world_size"])
            )
            total_iters  = int(opt["train"]["total_iter"])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.'
            )

        elif phase == "val":
            val_set    = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu = opt["num_gpu"],
                dist    = opt["dist"],
                sampler = None,
                seed    = opt["manual_seed"],
            )
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}'
            )
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main(args):
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    args["checkpoints_dir"] = mon.Path(args["checkpoints_dir"])
    args["checkpoints_dir"].mkdir(parents=True, exist_ok=True)

    # Automatic resume 
    state_folder_path = "experiments/{}/training_states/".format(args["name"])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = "{}.state".format(max([int(x[0:-6]) for x in states]))
        resume_state   = os.path.join(state_folder_path, max_state_file)
        args["path"]["resume_state"] = resume_state

    # load resume states if necessary
    if args["path"].get("resume_state"):
        device_id    = torch.cuda.current_device()
        resume_state = torch.load(
            args["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(args)
        if args["logger"].get("use_tb_logger") and "debug" not in args["name"] and args["rank"] == 0:
            mkdir_and_rename(osp.join("tb_logger", args["name"]))

    # Initialize loggers
    logger, tb_logger = init_loggers(args)

    # Create train and validation dataloaders
    result = create_train_val_dataloader(args, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # Create model
    if resume_state:  # resume training
        check_resume(args, resume_state["iter"])
        model = create_model(args)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch  = resume_state["epoch"]
        current_iter = resume_state["iter"]
    else:
        model        = create_model(args)
        start_epoch  = 0
        current_iter = 0

    # Create message logger (formatted outputs)
    msg_logger = MessageLogger(args, current_iter, tb_logger)

    # Dataloader prefetcher
    prefetch_mode = args["datasets"]["train"].get("prefetch_mode")
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, args)
        logger.info(f"Use {prefetch_mode} prefetch dataloader")
        if args["datasets"]["train"].get("pin_memory") is not True:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}. "
            f"Supported ones are: ``None``, ``'cuda'``, ``'cpu'``."
        )

    # Training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_time  = time.time()
    iter_time  = time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time
            
            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=args["train"].get("warmup_iter", -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % args["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % args["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if args.get("val") is not None and (current_iter % args["val"]["val_freq"] == 0):
                rgb2bgr   = args["val"].get("rgb2bgr", True)
                # Whether use uint8 image to compute metrics
                use_image = args["val"].get("use_image", True)
                model.validation(val_loader, current_iter, tb_logger, args["val"]["save_img"], rgb2bgr, use_image )

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
    if args.get("val") is not None:
        rgb2bgr   = args["val"].get("rgb2bgr",   True)
        use_image = args["val"].get("use_image", True)
        model.validation(val_loader, current_iter, tb_logger, args["val"]["save_img"], rgb2bgr, use_image)
    if tb_logger:
        tb_logger.close()


def parse_args(is_train: bool = True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt",        type=str, required=True,                              help="Path to option YAML file.")
    parser.add_argument("--launcher",   choices=["none", "pytorch", "slurm"], default="none", help="job launcher")
    parser.add_argument("--local-rank", type=int, default=0)
    args     = parser.parse_args()
    args.opt = "options/train" + f"{args.opt}"
    opt      = parse(args.opt, is_train=is_train)

    # Distributed settings
    if args.launcher == "none":
        opt["dist"] = False
        console.log("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)
            console.log("init dist .. ", args.launcher)

    opt["rank"], opt["world_size"] = get_dist_info()

    # Random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    return opt


if __name__ == "__main__":
    args = parse_args(is_train=True)
    main(args)
