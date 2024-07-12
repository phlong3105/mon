#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import socket
import time
from collections import OrderedDict
from warnings import warn

import click
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import my_test as test  # import test.py to get mAP after each epoch
from models.yolo import Model
import mon
from utils.datasets import create_dataloader
from utils.general import (
    check_dataset, check_file, check_git_status, check_img_size, fitness, fitness_ap,
    fitness_ap50, fitness_f1, fitness_p, fitness_r, get_latest_run, increment_path,
    init_seeds, labels_to_class_weights, labels_to_image_weights, print_mutation,
    set_logging, strip_optimizer,
)
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_evolution, plot_images, plot_labels, plot_results
from utils.torch_utils import intersect_dicts, ModelEMA, select_device, torch_distributed_zero_first

logger        = logging.getLogger(__name__)
console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


# region Train

def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f"Hyperparameters {hyp}")
    weights          = opt.weights
    weights          = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    save_dir         = mon.Path(opt.save_dir)
    epochs           = opt.epochs
    batch_size       = opt.batch_size
    total_batch_size = opt.total_batch_size
    rank             = opt.global_rank
    
    # Directories
    wdir         = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last         = wdir     / "last.pt"
    best         = wdir     / "best.pt"
    best_p       = wdir     / "best_p.pt"
    best_r       = wdir     / "best_r.pt"
    best_f1      = wdir     / "best_f1.pt"
    best_ap50    = wdir     / "best_ap50.pt"
    best_ap      = wdir     / "best_ap.pt"
    results_file = save_dir / "results.txt"
    
    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    
    # Configure
    plots = not opt.evolve  # Create plots
    cuda  = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data, encoding="utf-8") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        train_    = data_dict["train"]
        val_      = data_dict["val"]
        test_     = data_dict["test"]
        if isinstance(train_, list):
            train_ = [str(mon.DATA_DIR / t) for t in train_]
        elif train_:
            train_ = str(mon.DATA_DIR / train_)
        if isinstance(val_, list):
            val_   = [str(mon.DATA_DIR / t) for t in val_]
        elif val_:
            val_   = str(mon.DATA_DIR / val_)
        if isinstance(test_, list):
            test_  = [str(mon.DATA_DIR / t) for t in test_]
        elif test_:
            test_  = str(mon.DATA_DIR / test_)
        data_dict["train"] = train_
        data_dict["val"]   = val_
        data_dict["test"]  = test_

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path  = data_dict["val"]
    nc, names  = (1, ["item"]) if opt.single_cls else (int(data_dict["nc"]), data_dict["names"])    # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt       = torch.load(weights, map_location=device)  # load checkpoint
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])  # force autoanchor
        model      = Model(opt.model or ckpt["model"].yaml, ch=3, nc=nc).to(device)  # create
        exclude    = ["anchor"] if opt.model or hyp.get("anchors") else []  # exclude keys
        # state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = ckpt["model"]
        if not isinstance(state_dict, OrderedDict):
            state_dict = state_dict.float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info("Transferred %g/%g items from %s" % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.model, ch=3, nc=nc).to(device)  # create
    
    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False
            
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)          # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, "im"):
            for iv in v.im:
                pg0.append(iv.implicit)
        if hasattr(v, "ia"):
            for iv in v.ia:
                pg0.append(iv.implicit)
        if hasattr(v, "id"):
            for iv in v.id:
                pg0.append(iv.implicit)
        if hasattr(v, "iq"):
            for iv in v.iq:
                pg0.append(iv.implicit)
        if hasattr(v, "ix"):
            for iv in v.ix:
                pg0.append(iv.implicit)
        if hasattr(v, "ie"):
            for iv in v.ie:
                pg0.append(iv.implicit)
        if hasattr(v, "ic"):
            pg0.append(v.ic.implicit)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    logger.info("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf        = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Logging
    if wandb and wandb.run is None:
        opt.hyp   = hyp  # add hyperparameters
        wandb_run = wandb.init(
            config  = opt,
            resume  = "allow",
            project = "YOLOR" if opt.project == "runs/train" else mon.Path(opt.project).stem,
            name    = save_dir.stem,
            id      = ckpt.get("wandb_id") if "ckpt" in locals() else None
        )

    # Resume
    start_epoch       = 0
    best_fitness      = 0.0
    best_fitness_p    = 0.0
    best_fitness_r    = 0.0
    best_fitness_f1   = 0.0
    best_fitness_ap50 = 0.0
    best_fitness_ap   = 0.0
    
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness      = ckpt.get("best_fitness",      0.0)
            best_fitness_p    = ckpt.get("best_fitness_p",    0.0)
            best_fitness_r    = ckpt.get("best_fitness_r",    0.0)
            best_fitness_f1   = ckpt.get("best_fitness_f1",   0.0)
            best_fitness_ap50 = ckpt.get("best_fitness_ap50", 0.0)
            best_fitness_ap   = ckpt.get("best_fitness_ap",   0.0)
        
        # Results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt
        
        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if opt.resume:
            assert start_epoch > 0, "%s training to %g epochs is finished, nothing to resume." % (weights, epochs)
        if epochs < start_epoch:
            logger.info("%s has been trained for %g epochs. Fine-tuning for %g additional epochs." % (weights, ckpt["epoch"], epochs))
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = 64  # int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.imgsz]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    dataloader, dataset = create_dataloader(
        path       = train_path,
        imgsz      = imgsz,
        batch_size = batch_size,
        stride     = gs,
        opt        = opt,
        hyp        = hyp,
        augment    = True,
        cache      = opt.cache_images,
        rect       = opt.rect,
        rank       = rank,
        world_size = opt.world_size,
        workers    = opt.workers,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb  = len(dataloader)  # number of batches
    assert mlc < nc, "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (mlc, nc, opt.data, nc - 1)
    
    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader  = create_dataloader(
            path       = test_path,
            imgsz      = imgsz_test,
            batch_size = batch_size*2,
            stride     = gs,
            opt        = opt,
            hyp        = hyp,
            cache      = opt.cache_images and not opt.notest,
            rect       = True,
            rank       = -1,
            world_size = opt.world_size,
            workers    = opt.workers
        )[0]  # testloader

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c      = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.0  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, save_dir=save_dir)
                if tb_writer:
                    tb_writer.add_histogram("classes", c, 0)
                if wandb:
                    wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob("*labels*.png")]})

            # Anchors
            # if not opt.noautoanchor:
            #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    hyp["cls"] *= nc / 80.0  # scale coco-tuned hyp['cls'] to current dataset
    model.nc    = nc         # attach number of classes to model
    model.hyp   = hyp        # attach hyperparameters to model
    model.gr    = 1.0        # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Start training
    t0      = time.time()
    nw      = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps    = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler  = amp.GradScaler(enabled=cuda)
    logger.info(
        "Image sizes %g train, %g test\n"
        "Using %g dataloader workers\nLogging results to %s\n"
        "Starting training for %g epochs..." % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs)
    )
    
    torch.save(model, wdir / "init.pt")
    
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        
        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size"))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni   = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns   = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem   = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s     = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f"train_batch{ni}.jpg"  # filename
                    plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob("train*.jpg")]})
        
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                if epoch >= 0:
                    results, maps, times = test.test(
                        opt        = opt,
                        data       = opt.data,
                        batch_size = batch_size * 2,
                        imgsz      = imgsz_test,
                        conf_thres = opt.conf,
                        iou_thres  = opt.iou,
                        max_det    = opt.max_det,
                        single_cls = opt.single_cls,
                        model      = ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                        dataloader = testloader,
                        save_dir   = save_dir,
                        plots      = plots and final_epoch,
                        log_imgs   = opt.log_imgs if wandb else 0,
                    )

            # Update best mAP
            fi      = fitness(np.array(results).reshape(1, -1))         # weighted combination of [P, R, F1, mAP@0.5, mAP@0.5:0.95]
            fi_p    = fitness_p(np.array(results).reshape(1, -1))       # weighted combination of [P, R, F1, mAP@0.5, mAP@0.5:0.95]
            fi_r    = fitness_r(np.array(results).reshape(1, -1))       # weighted combination of [P, R, F1, mAP@0.5, mAP@0.5:0.95]
            if (fi_p > 0.0) or (fi_r > 0.0):
                fi_f1 = fitness_f1(np.array(results).reshape(1, -1))    # weighted combination of [P, R, F1, mAP@0.5, mAP@0.5:0.95]
            else:
                fi_f1 = 0.0
            fi_ap50 = fitness_ap50(np.array(results).reshape(1, -1))    # weighted combination of [P, R, F1, mAP@0.5, mAP@0.5:0.95]
            fi_ap   = fitness_ap(np.array(results).reshape(1, -1))      # weighted combination of [P, R, F1, mAP@0.5, mAP@0.5:0.95]
            
            if fi > best_fitness:
                best_fitness      = fi
            if fi_p > best_fitness_p:
                best_fitness_p    = fi_p
            if fi_r > best_fitness_r:
                best_fitness_r    = fi_r
            if fi_f1 > best_fitness_f1:
                best_fitness_f1   = fi_f1
            if fi_ap50 > best_fitness_ap50:
                best_fitness_ap50 = fi_ap50
            if fi_ap > best_fitness_ap:
                best_fitness_ap   = fi_ap
            
            # Write
            with open(results_file, "a") as f:
                f.write(s + "%10.4g" * 7 % results + "\n")  # P, R, F1, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system("gsutil cp %s gs://%s/results/results%s.txt" % (results_file, opt.bucket, opt.name))
            
            # Log
            tags = [
                "train/box_loss",
                "train/obj_loss",
                "train/cls_loss",  # train loss
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/f1(B)",
                "metrics/map@0.5(B)",
                "metrics/map@0.5-0.95(B)",
                "val/box_loss",
                "val/obj_loss",
                "val/cls_loss",  # val loss
                "x/lr0",
                "x/lr1",
                "x/lr2"
            ]  # params
            results = list(results)
            results.insert(2, fi_f1)
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, "r") as f:  # create checkpoint
                    ckpt = {
                        "epoch"            : epoch,
                        "best_fitness"     : best_fitness,
                        "best_fitness_p"   : best_fitness_p,
                        "best_fitness_r"   : best_fitness_r,
                        "best_fitness_f1"  : best_fitness_f1,
                        "best_fitness_ap50": best_fitness_ap50,
                        "best_fitness_ap"  : best_fitness_ap,
                        "training_results" : f.read(),
                        "config"           : opt.model,
                        "nc"               : nc,
                        "model"            : ema.ema.module.state_dict() if hasattr(ema, "module") else ema.ema.state_dict(),
                        "optimizer"        : None if final_epoch else optimizer.state_dict(),
                        "wandb_id"         : wandb_run.id if wandb else None,
                    }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                # if (best_fitness == fi) and (epoch >= 200):
                #     torch.save(ckpt, wdir / "best_{:03d}.pt".format(epoch))
                # if best_fitness == fi:
                #     torch.save(ckpt, wdir / "best_overall.pt")
                if best_fitness_p == fi_p:
                    torch.save(ckpt, best_p)
                if best_fitness_r == fi_r:
                    torch.save(ckpt, best_r)
                if best_fitness_f1 == fi_f1:
                    torch.save(ckpt, best_f1)
                if best_fitness_ap50 == fi_ap50:
                    torch.save(ckpt, best_ap50)
                if best_fitness_ap == fi_ap:
                    torch.save(ckpt, best_ap)
                # if epoch == 0:
                #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                # if ((epoch+1) % 25) == 0:
                #     torch.save(ckpt, wdir / "epoch_{:03d}.pt".format(epoch))
                # if epoch >= (epochs-5):
                #     torch.save(ckpt, wdir / "last_{:03d}.pt".format(epoch))
                # elif epoch >= 420:
                #     torch.save(ckpt, wdir / "last_{:03d}.pt".format(epoch))
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = opt.name if opt.name.isnumeric() else ""
        fresults   = save_dir / f"results{n}.txt"
        flast      =  wdir    / f"last{n}.pt"
        fbest      =  wdir    / f"best{n}.pt"
        fbest_p    =  wdir    / f"best_p{n}.pt"
        fbest_r    =  wdir    / f"best_r{n}.pt"
        fbest_f1   =  wdir    / f"best_f1{n}.pt"
        fbest_ap50 =  wdir    / f"best_ap50{n}.pt"
        fbest_ap   =  wdir    / f"best_ap{n}.pt"
        for f1, f2 in zip(
            [last,  best,  best_p,  best_r,  best_f1,  best_ap,    best_ap,  results_file],
            [flast, fbest, fbest_p, fbest_r, fbest_f1, fbest_ap50, fbest_ap, fresults]
        ):
            if f1.exists():
                os.rename(f1, f2)  # rename
                if str(f2).endswith(".pt"):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
                    os.system("gsutil cp %s gs://%s/weights" % (f2, opt.bucket)) if opt.bucket else None  # upload
        # Finish
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                wandb.log({"Results": [wandb.Image(str(save_dir / x), caption=x) for x in ["results.png", "precision-recall_curve.png"]]})
        print("%g epochs completed in %.3f hours.\n" % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--arch",       type=str, default=None, help="Model architecture.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--project",    type=str, default=None, help="Project name.")
@click.option("--variant",    type=str, default=None, help="Variant name.")
@click.option("--fullname",   type=str, default=None, help="Fullname to save the model's weight.")
@click.option("--save-dir",   type=str, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--device",     type=str,   default=None, help="Running devices.")
@click.option("--local-rank", type=int,   default=-1,   help="DDP parameter, do not modify.")
@click.option("--epochs",     type=int,   default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int,   default=None, help="Stop training once this number of steps is reached.")
@click.option("--conf",       type=float, default=None, help="Confidence threshold.")
@click.option("--iou",        type=float, default=None, help="IoU threshold.")
@click.option("--max-det",    type=int,   default=None, help="Max detections per image.")
@click.option("--exist-ok",   is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    config    : str,
    arch      : str,
    model     : str,
    root      : str,
    project   : str,
    variant   : str,
    fullname  : str,
    save_dir  : str,
    weights   : str,
    local_rank: int,
    device    : str,
    epochs    : int,
    steps     : int,
    conf      : float,
    iou       : float,
    max_det   : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    model    = model    or args.get("model")
    data     = args.get("data")
    root     = root     or args.get("root")
    project  = project  or args.get("project")
    variant  = variant  or args.get("variant")
    fullname = fullname or args.get("name")
    weights  = weights  or args.get("weights")
    device   = device   or args.get("device")
    hyp      = args.get("hyp")
    epochs   = epochs   or args.get("epochs")
    conf     = conf     or args.get("conf")
    iou      = iou      or args.get("iou")
    max_det  = max_det  or args.get("max_det")
    exist_ok = exist_ok or args.get("exist_ok")
    verbose  = verbose  or args.get("verbose")
    
    # Parse arguments
    model    = mon.Path(model)
    model    = model if model.exists() else _current_dir / "config" / model.name
    model    = str(model.config_file())
    data     = mon.Path(data)
    data     = data  if data.exists() else _current_dir / "data" / data.name
    data     = str(data.config_file())
    root     = mon.Path(root)
    project  = root.name or project
    save_dir = save_dir  or root / "run" / "train" / fullname
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    hyp      = mon.Path(hyp)
    hyp      = hyp if hyp.exists() else _current_dir / "data" / hyp.name
    hyp      = hyp.yaml_file()
    
    # Update arguments
    args["config"]     = config
    
    args["root"]       = root
    
    args["weights"]    = weights
    args["model"]      = model
    args["data"]       = data
    args["root"]       = root
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
    args["device"]     = device
    args["local_rank"] = local_rank
    args["hyp"]        = str(hyp)
    args["epochs"]     = epochs
    args["steps"]      = steps
    args["conf"]       = conf
    args["iou"]        = iou
    args["max_det"]    = max_det
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    
    opt = argparse.Namespace(**args)
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(opt.save_dir))
    mon.Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size       = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else  1
    opt.global_rank      = int(os.environ["RANK"])       if "RANK"       in os.environ else -1
    
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
    
    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(mon.Path(ckpt).parent.parent / "opt.yaml") as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.model, opt.weights, opt.resume = "", ckpt, True
        logger.info("Resuming training from %s" % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data     = check_file(opt.data)   # check files
        opt.model    = check_file(opt.model)  # check files
        opt.hyp      = check_file(opt.hyp)    # check files
        assert len(opt.model) or len(opt.weights), "either --config or --weights must be specified"
        opt.imgsz    = mon.to_list(opt.imgsz)
        opt.imgsz.extend([opt.imgsz[-1]] * (2 - len(opt.imgsz)))  # extend to 2 sizes (train, test)
        opt.name     = "evolve" if opt.evolve else opt.name
        opt.save_dir = increment_path(mon.Path(opt.save_dir), exist_ok=opt.exist_ok | opt.evolve)  # increment run
    
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")  # distributed backend
        assert opt.batch_size % opt.world_size == 0, "--batch-size must be multiple of CUDA device count"
        opt.batch_size = opt.total_batch_size // opt.world_size
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120')
            )
            hyp["box"] = hyp.pop("giou")
    
    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer, wandb)
    
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0"            : (1, 1e-5, 1e-1),   # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf"            : (1, 0.01, 1.0),    # final OneCycleLR learning rate (lr0 * lrf)
            "momentum"       : (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay"   : (1, 0.0, 0.001),   # optimizer weight decay
            "warmup_epochs"  : (1, 0.0, 5.0),     # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),    # warmup initial momentum
            "warmup_bias_lr" : (1, 0.0, 0.2),     # warmup initial bias lr
            "box"            : (1, 0.02, 0.2),    # box loss gain
            "cls"            : (1, 0.2, 4.0),     # cls loss gain
            "cls_pw"         : (1, 0.5, 2.0),     # cls BCELoss positive_weight
            "obj"            : (1, 0.2, 4.0),     # obj loss gain (scale with pixels)
            "obj_pw"         : (1, 0.5, 2.0),     # obj BCELoss positive_weight
            "iou_t"          : (0, 0.1, 0.7),     # IoU training threshold
            "anchor_t"       : (1, 2.0, 8.0),     # anchor-multiple threshold
            "anchors"        : (2, 2.0, 10.0),    # anchors per output grid (0 to ignore)
            "fl_gamma"       : (0, 0.0, 2.0),     # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h"          : (1, 0.0, 0.1),     # image HSV-Hue augmentation (fraction)
            "hsv_s"          : (1, 0.0, 0.9),     # image HSV-Saturation augmentation (fraction)
            "hsv_v"          : (1, 0.0, 0.9),     # image HSV-Value augmentation (fraction)
            "degrees"        : (1, 0.0, 45.0),    # image rotation (+/- deg)
            "translate"      : (1, 0.0, 0.9),     # image translation (+/- fraction)
            "scale"          : (1, 0.0, 0.9),     # image scale (+/- gain)
            "shear"          : (1, 0.0, 10.0),    # image shear (+/- deg)
            "perspective"    : (0, 0.0, 0.001),   # image perspective (+/- fraction), range 0-0.001
            "flipud"         : (1, 0.0, 1.0),     # image flip up-down (probability)
            "fliplr"         : (0, 0.0, 1.0),     # image flip left-right (probability)
            "mosaic"         : (1, 0.0, 1.0),     # image mixup (probability)
            "mixup"          : (1, 0.0, 1.0),     # image mixup (probability)
        }
        
        assert opt.local_rank == -1, "DDP mode not implemented for --evolve"
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = mon.Path(opt.save_dir) / "hyp_evolved.yaml"  # save best result here
        if opt.bucket:
            os.system("gsutil cp gs://%s/evolve.txt ." % opt.bucket)  # download evolve.txt if exists
        
        for _ in range(300):  # generations to evolve
            if mon.Path("evolve.txt").exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: 'single' or 'weighted'
                x      = np.loadtxt("evolve.txt", ndmin=2)
                n      = min(5, len(x))  # number of previous results to consider
                x      = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w      = fitness(x) - fitness(x).min()   # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]    # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()     # weighted combination
                
                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr   = np.random
                npr.seed(int(time.time()))
                g     = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng    = len(meta)
                v     = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):   # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate
            
            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)   # significant digits
            
            # Train mutation
            results = train(hyp.copy(), opt, device, wandb=wandb)
            
            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
        
        # Plot results
        plot_evolution(yaml_file)
        print(
            f"Hyperparameter evolution complete. Best results saved as: {yaml_file}\n"
            f"Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}"
        )
        
        return str(opt.save_dir)
        

if __name__ == "__main__":
    main()

# endregion
