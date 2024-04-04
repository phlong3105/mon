#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import socket

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import config.options as option
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from models import create_model
import mon
from utils import util

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def init_dist(backend="nccl", **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")
    rank     = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def train(args: argparse.Namespace):
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    save_dir = mon.Path(args.save_dir)
    device   = args.device
    launcher = args.launcher
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Override options with args
    opt = option.parse(args.opt, is_train=True)

    experiments_root = save_dir
    opt["path"]["experiments_root"] = str(experiments_root)
    opt["path"]["models"]           = str(experiments_root / "weights")
    opt["path"]["training_state"]   = str(experiments_root / "training_state")
    opt["path"]["log"]              = str(experiments_root)
    opt["path"]["tb_logger"]        = str(experiments_root / "tb_logger")
    opt["path"]["val_images"]       = str(experiments_root / "val_images")
    opt["path"]["pretrain_model_G"] = str(weights)
    
    opt["datasets"]["train"]["dataroot_GT"] = mon.DATA_DIR / opt["datasets"]["train"]["dataroot_GT"]
    opt["datasets"]["train"]["dataroot_LQ"] = mon.DATA_DIR / opt["datasets"]["train"]["dataroot_LQ"]
    opt["datasets"]["val"]["dataroot_GT"]   = mon.DATA_DIR / opt["datasets"]["val"]["dataroot_GT"]
    opt["datasets"]["val"]["dataroot_LQ"]   = mon.DATA_DIR / opt["datasets"]["val"]["dataroot_LQ"]
    
    opt["device"] = device
    
    # Distributed training settings
    if launcher == "none":  # disabled distributed training
        opt["dist"] = False
        rank        = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        world_size  = torch.distributed.get_world_size()
        rank        = torch.distributed.get_rank()
    
    # Loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id    = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None
    
    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(opt["path"]["experiments_root"])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt["path"].items() if not key == "experiments_root" and "pretrain_model" not in key and "resume" not in key))
        
        # config loggers. Before it, the log will not work
        util.setup_logger("base", opt["path"]["log"], "train_" + opt["name"], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info("You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"] + "/" + opt["name"])
    else:
        util.setup_logger("base", opt["path"]["log"], "train", level=logging.INFO, screen=True)
        logger = logging.getLogger("base")

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # Random seed
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info("Random seed: {}".format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # Create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)

            # print(train_set[0])
            # import pdb; pdb.set_trace()
            
            train_size   = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters  = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs  = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info("Number of train images: {:,d}, iters: {:,d}".format(len(train_set), train_size))
                logger.info("Total epochs needed: {:d} for iters {:,d}".format(total_epochs, total_iters))
        elif phase == "val":
            val_set    = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info("Number of val images in [{:s}]: {:d}".format(dataset_opt["name"], len(val_set)))
        else:
            pass
            # raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)

    # Resume training
    if resume_state:
        logger.info("Resuming training from epoch: {}, iter: {}.".format(resume_state["epoch"], resume_state["iter"]))
        start_epoch  = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch  = 0

    # Training
    logger.info("Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # log
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "[epoch:{:3d}, iter:{:8,d}, lr:(".format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += "{:.3e},".format(v)
                message += ")] "
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            '''
            if opt["datasets"].get("val", None) and current_step % opt["train"]["val_freq"] == 0:
                if opt["model"] in ["sr", "srgan"] and rank <= 0:  # image restoration validation
                    # does not support multi-GPU validation
                    pbar     = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.0
                    idx      = 0
                    for val_data in val_loader:
                        idx      += 1
                        img_name  = os.path.splitext(os.path.basename(val_data["LQ_path"][0]))[0]
                        img_dir   = os.path.join(opt["path"]["val_images"], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img  = util.tensor2img(visuals["rlt"])  # uint8
                        gt_img  = util.tensor2img(visuals["GT"])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir, "{:s}_{:d}.png".format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img  = util.crop_border([sr_img, gt_img], opt["scale"])
                        avg_psnr       += util.calculate_psnr(sr_img, gt_img)
                        pbar.update("Test {}".format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info("# Validation # PSNR: {:.4e}".format(avg_psnr))
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("psnr", avg_psnr, current_step)
                else:  # video restoration validation
                    if opt["dist"]:
                        # multi-GPU testing
                        psnr_rlt = {}  # with border and center frames
                        if rank == 0:
                            pbar = util.ProgressBar(len(val_set))

                        random_index = random.randint(0, len(val_set)-1)
                        for idx in range(rank, len(val_set), world_size):

                            if not (idx == random_index):
                                continue

                            val_data = val_set[idx]
                            val_data["LQs"].unsqueeze_(0)
                            val_data["GT"].unsqueeze_(0)
                            folder = val_data["folder"]
                            idx_d, max_idx = val_data["idx"].split("/")
                            idx_d, max_idx = int(idx_d), int(max_idx)
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32, device="cuda")
                            # tmp = torch.zeros(max_idx, dtype=torch.float32, device="cuda")
                            model.feed_data(val_data)
                            model.test()
                            visuals  = model.get_current_visuals()
                            sou_img  = util.tensor2img(visuals["LQ"])
                            rlt_img  = util.tensor2img(visuals["rlt"])  # uint8
                            rlt_img2 = util.tensor2img(visuals["rlt2"])  # uint8
                            gt_img   = util.tensor2img(visuals["GT"])  # uint8
                            ill_img  = util.tensor2img(visuals["ill"])
                            rlt_img3 = util.tensor2img(visuals["rlt3"])

                            save_img = np.concatenate([sou_img, rlt_img, ill_img, gt_img, rlt_img3, rlt_img2], axis=0)
                            im_path  = os.path.join(opt["path"]["val_images"], "%06d.png" % current_step)
                            cv2.imwrite(im_path, save_img.astype(np.uint8))

                            # calculate PSNR
                            psnr_rlt[folder][idx_d] = util.calculate_psnr(rlt_img, gt_img)
                        # # collect data
                        for _, v in psnr_rlt.items():
                            dist.reduce(v, 0)
                        dist.barrier()

                        if rank == 0:
                            psnr_rlt_avg = {}
                            psnr_total_avg = 0.
                            for k, v in psnr_rlt.items():
                                psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                                psnr_total_avg += psnr_rlt_avg[k]
                            psnr_total_avg /= len(psnr_rlt)
                            log_s = "# Validation # PSNR: {:.4e}:".format(psnr_total_avg)
                            for k, v in psnr_rlt_avg.items():
                                log_s += " {}: {:.4e}".format(k, v)
                            logger.info(log_s)
                            if opt["use_tb_logger"] and "debug" not in opt["name"]:
                                tb_logger.add_scalar("psnr_avg", psnr_total_avg, current_step)
                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)
                    else:
                        pbar           = util.ProgressBar(len(val_loader))
                        psnr_rlt       = {}  # with border and center frames
                        psnr_rlt_avg   = {}
                        psnr_total_avg = 0.0
                        for val_data in val_loader:
                            folder = val_data["folder"][0]
                            idx_d  = val_data["idx"].item()
                            # border = val_data["border"].item()
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = []

                            model.feed_data(val_data)
                            model.test()
                            visuals = model.get_current_visuals()
                            rlt_img = util.tensor2img(visuals["rlt"])  # uint8
                            gt_img  = util.tensor2img(visuals["GT"])  # uint8

                            # calculate PSNR
                            psnr = util.calculate_psnr(rlt_img, gt_img)
                            psnr_rlt[folder].append(psnr)
                            pbar.update("Test {} - {}".format(folder, idx_d))
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = sum(v) / len(v)
                            psnr_total_avg += psnr_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        log_s = "# Validation # PSNR: {:.4e}:".format(psnr_total_avg)
                        for k, v in psnr_rlt_avg.items():
                            log_s += " {}: {:.4e}".format(k, v)
                        logger.info(log_s)
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            tb_logger.add_scalar("psnr_avg", psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
            '''
            
            # save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of training.")
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
@click.option("--launcher",   type=click.Choice(["none", "pytorch"]), default="none", help="Job launcher.")
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
    
    # Prioritize input args --> config file args
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
