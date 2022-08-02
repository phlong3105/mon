#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLOv5 🚀 by Ultralytics, GPL-3.0 license
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from munch import Munch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.optim import SGD
from tqdm import tqdm

import one.vision.detection.yolov5_v6_1.val as val  # for end-of-epoch mAP
from one.constants import DATA_DIR
from one.constants import PRETRAINED_DIR
from one.vision.detection.yolov5_v6_1.models.experimental import attempt_load
from one.vision.detection.yolov5_v6_1.models.yolo import Model
from one.vision.detection.yolov5_v6_1.utils.autoanchor import check_anchors
from one.vision.detection.yolov5_v6_1.utils.autobatch import \
    check_train_batch_size
from one.vision.detection.yolov5_v6_1.utils.callbacks import Callbacks
from one.vision.detection.yolov5_v6_1.utils.datasets import create_dataloader
from one.vision.detection.yolov5_v6_1.utils.downloads import attempt_download
from one.vision.detection.yolov5_v6_1.utils.general import check_dataset
from one.vision.detection.yolov5_v6_1.utils.general import check_file
from one.vision.detection.yolov5_v6_1.utils.general import check_git_status
from one.vision.detection.yolov5_v6_1.utils.general import check_img_size
from one.vision.detection.yolov5_v6_1.utils.general import check_requirements
from one.vision.detection.yolov5_v6_1.utils.general import check_suffix
from one.vision.detection.yolov5_v6_1.utils.general import check_yaml
from one.vision.detection.yolov5_v6_1.utils.general import colorstr
from one.vision.detection.yolov5_v6_1.utils.general import get_latest_run
from one.vision.detection.yolov5_v6_1.utils.general import increment_path
from one.vision.detection.yolov5_v6_1.utils.general import init_seeds
from one.vision.detection.yolov5_v6_1.utils.general import intersect_dicts
from one.vision.detection.yolov5_v6_1.utils.general import \
    labels_to_class_weights
from one.vision.detection.yolov5_v6_1.utils.general import \
    labels_to_image_weights
from one.vision.detection.yolov5_v6_1.utils.general import LOGGER
from one.vision.detection.yolov5_v6_1.utils.general import methods
from one.vision.detection.yolov5_v6_1.utils.general import one_cycle
from one.vision.detection.yolov5_v6_1.utils.general import print_args
from one.vision.detection.yolov5_v6_1.utils.general import print_mutation
from one.vision.detection.yolov5_v6_1.utils.general import strip_optimizer
from one.vision.detection.yolov5_v6_1.utils.loggers import Loggers
from one.vision.detection.yolov5_v6_1.utils.loggers.wandb.wandb_utils import \
    check_wandb_resume
from one.vision.detection.yolov5_v6_1.utils.loss import ComputeLoss
from one.vision.detection.yolov5_v6_1.utils.metrics import fitness
from one.vision.detection.yolov5_v6_1.utils.plots import plot_evolve
from one.vision.detection.yolov5_v6_1.utils.plots import plot_labels
from one.vision.detection.yolov5_v6_1.utils.torch_utils import de_parallel
from one.vision.detection.yolov5_v6_1.utils.torch_utils import EarlyStopping
from one.vision.detection.yolov5_v6_1.utils.torch_utils import ModelEMA
from one.vision.detection.yolov5_v6_1.utils.torch_utils import select_device
from one.vision.detection.yolov5_v6_1.utils.torch_utils import \
    torch_distributed_zero_first

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK       = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


# H1: - Functional -------------------------------------------------------------

def train(
    hyp,  # path/to/hyp.yaml or hyp dictionary
    args: dict | Munch | argparse.Namespace,
    device,
    callbacks
):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
    
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(args.save_dir), args.epochs, args.batch_size, args.weights, args.single_cls, args.evolve, args.data, args.cfg, \
        args.resume, args.noval, args.nosave, args.workers, args.freeze

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))

    # Save run settings
    if not evolve:
        with open(save_dir / "hyp.yaml", "w") as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / "args.yaml", "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, args, hyp, LOGGER)  # loggers measurement
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = args.weights, args.epochs, args.hyp, args.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda  = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    
    # train_path, val_path = data_dict["train"], data_dict["val"]
    if os.path.isdir(DATA_DIR):
        train_path = os.path.join(DATA_DIR, data_dict["train"])
        val_path   = os.path.join(DATA_DIR, data_dict["val"])
    else:
        train_path = os.path.join(data_dict["path"], data_dict["train"])
        val_path   = os.path.join(data_dict["path"], data_dict["val"])
    
    nc    = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = ["item"] if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {data}"  # check
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt    = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        model   = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd     = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd     = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs    = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs                  = 64  # nominal batch size
    accumulate           = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if args.optimizer == "Adam":
        optimizer = Adam(g0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
    elif args.optimizer == "AdamW":
        optimizer = AdamW(g0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group({"params": g1, "weight_decay": hyp["weight_decay"]})  # add g1 with weight_decay
    optimizer.add_param_group({"params": g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if args.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp["lrf"]
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # EMA
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if resume:
            assert start_epoch > 0, f"{weights} training to {epochs} epochs is finished, nothing to resume."
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning("WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
                       "See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.")
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if args.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        path          = train_path,
        imgsz         = imgsz,
        batch_size    = batch_size // WORLD_SIZE,
        stride        = gs,
        single_cls    = single_cls,
        hyp           = hyp,
        augment       = True,
        cache         = None if args.cache == "val" else args.cache,
        rect          = args.rect,
        rank          = LOCAL_RANK,
        workers       = workers,
        image_weights = args.image_weights,
        quad          = args.quad,
        prefix        = colorstr("train: "),
        shuffle       = True
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb  = len(train_loader)  # number of batches
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            path       = val_path,
            imgsz      = imgsz,
            batch_size = batch_size // WORLD_SIZE * 2,
            stride     = gs,
            single_cls = single_cls,
            hyp        = hyp,
            cache      = None if noval else args.cache,
            rect       = True,
            rank       = -1,
            workers    = workers * 2,
            pad        = 0.5,
            prefix     = colorstr("val: ")
        )[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not args.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end")

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes
    nl                      = de_parallel(model).model[-1].nl  # number of measurement layers (to scale hyps)
    hyp["box"]             *= 3 / nl  # scale to layers
    hyp["cls"]             *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"]             *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"]  = args.label_smoothing
    model.nc                = nc   # attach number of classes to model
    model.hyp               = hyp  # attach hyperparameters to model
    model.class_weights     = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names             = names

    # Start training
    t0            = time.time()
    nw            = max(round(hyp["warmup_epochs"] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps          = np.zeros(nc)  # mAP per class
    results       = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler        = amp.GradScaler(enabled=cuda)
    stopper       = EarlyStopping(patience=args.patience)
    compute_loss  = ComputeLoss(model)  # init loss class
    LOGGER.info(f"Image sizes {imgsz} train, {imgsz} val\n"
                f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f"Starting training for {epochs} epochs...")
    
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights (optional, single-GPU only)
        if args.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%10s" * 7) % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size"))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni   = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns   = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred             = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if args.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem   = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%10s" * 2 + "%10.4g" * 5) % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", ni, model, imgs, targets, paths, plots, args.sync_bn)
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(
                    data_dict,
                    batch_size   = batch_size // WORLD_SIZE * 2,
                    imgsz        = imgsz,
                    model        = ema.ema,
                    single_cls   = single_cls,
                    dataloader   = val_loader,
                    save_dir     = save_dir,
                    plots        = False,
                    callbacks    = callbacks,
                    compute_loss = compute_loss
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch"       : epoch,
                    "best_fitness": best_fitness,
                    "model"       : deepcopy(de_parallel(model)).half(),
                    "ema"         : deepcopy(ema.ema).half(),
                    "updates"     : ema.updates,
                    "optimizer"   : optimizer.state_dict(),
                    "wandb_id"    : loggers.wandb.wandb_run.id if loggers.wandb else None,
                    "date"        : datetime.now().isoformat()
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (args.save_period > 0) and (epoch % args.save_period == 0):
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = val.run(
                        data_dict,
                        batch_size   = batch_size // WORLD_SIZE * 2,
                        imgsz        = imgsz,
                        model        = attempt_load(f, device).half(),
                        iou_thres    = 0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        single_cls   = single_cls,
                        dataloader   = val_loader,
                        save_dir     = save_dir,
                        save_json    = is_coco,
                        verbose      = True,
                        plots        = True,
                        callbacks    = callbacks,
                        compute_loss = compute_loss
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


# H1: - Main -------------------------------------------------------------------

def parse_args(known=False):
    weights_dir = PRETRAINED_DIR / "yolov5_v6_1"
    parser      = argparse.ArgumentParser()
    parser.add_argument("--weights",         default=weights_dir/"yolov5x6_v6_1_coco.pt",   type=str,                                   help="initial weights path")
    parser.add_argument("--cfg",             default="",                                    type=str,                                   help="model.yaml path")
    parser.add_argument("--data",            default=ROOT/"data/aic22retail.yaml",          type=str,                                   help="dataset.yaml path")
    parser.add_argument("--hyp",             default=ROOT/"data/hyps/hyp.scratch-low.yaml", type=str,                                   help="hyperparameters path")
    parser.add_argument("--epochs",          default=200,                                   type=int)
    parser.add_argument("--batch-size",      default=16,                                    type=int,                                   help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz",           default=1536,                                  type=int,                                   help="train, val image size (pixels)")
    parser.add_argument("--rect",            default=False,                                 action="store_true",                        help="rectangular training")
    parser.add_argument("--resume",          const=False,                                   nargs="?",                                  help="resume most recent training")
    parser.add_argument("--evolve",          const=300,                                     type=int, nargs="?",                        help="evolve hyperparameters for x generations")
    parser.add_argument("--bucket",          default="",                                    type=str,                                   help="gsutil bucket")
    parser.add_argument("--cache",           const="ram",                                   type=str, nargs="?",                        help="--cache images in 'ram' (default) or 'disk'")
    parser.add_argument("--image-weights",   default=False,                                 action="store_true",                        help="use weighted image selection for training")
    parser.add_argument("--device",          default="",                                                                                help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale",     default=False,                                 action="store_true",                        help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls",      default=False,                                 action="store_true",                        help="train multi-class data as single-class")
    parser.add_argument("--optimizer",       default="SGD",                                 type=str, choices=["SGD", "Adam", "AdamW"], help="optimizer")
    parser.add_argument("--sync-bn",         default=False,                                 action="store_true",                        help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers",         default=8,                                     type=int,                                   help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project",         default=ROOT/"runs/aic22retail",                                                           help="save to project/name")
    parser.add_argument("--name",            default="yolov5x6_v6_1_aic22retail_1536",                                                  help="save to project/name")
    parser.add_argument("--exist-ok",        default=False,                                 action="store_true",                        help="existing project/name ok, do not increment")
    parser.add_argument("--quad",            default=False,                                 action="store_true",                        help="quad dataloader")
    parser.add_argument("--cos-lr",          default=False,                                 action="store_true",                        help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", default=0.0,                                   type=float,                                 help="Label smoothing epsilon")
    parser.add_argument("--patience",        default=100,                                   type=int,                                   help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze",          default=[0],                                   nargs="+", type=int,                        help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period",     default=-1,                                    type=int,                                   help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--local_rank",      default=-1,                                    type=int,                                   help="DDP parameter, do not modify")
    parser.add_argument("--nosave",          default=False,                                 action="store_true",                        help="only save final checkpoint")
    parser.add_argument("--noval",           default=False,                                 action="store_true",                        help="only validate final epoch")
    parser.add_argument("--noautoanchor",    default=False,                                 action="store_true",                        help="disable AutoAnchor")
    
    # Weights & Biases arguments
    parser.add_argument("--entity",          default=None,                                                                              help="W&B: Entity")
    parser.add_argument("--upload_dataset",  default=False,                                 nargs="?", const=True,                      help="W&B: Upload data, 'val' option")
    parser.add_argument("--bbox_interval",   default=-1,                                    type=int,                                   help="W&B: Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias",  default="latest",                              type=str,                                   help="W&B: Version of dataset artifact to use")
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def main(args: dict | Munch | argparse.Namespace, callbacks=Callbacks()):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
    
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, args)
        check_git_status()
        check_requirements(exclude=["thop"])

    # Resume
    if args.resume and not check_wandb_resume(args) and not args.evolve:  # resume an interrupted run
        ckpt = args.resume if isinstance(args.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(Path(ckpt).parent.parent / "args.yaml", errors="ignore") as f:
            args = argparse.Namespace(**yaml.safe_load(f))  # replace
        args.cfg, args.weights, args.resume = "", ckpt, True  # reinstate
        LOGGER.info(f"Resuming training from {ckpt}")
    else:
        args.data, args.cfg, args.hyp, args.weights, args.project = \
            check_file(args.data), check_yaml(args.cfg), check_yaml(args.hyp), str(args.weights), str(args.project)  # checks
        assert len(args.cfg) or len(args.weights), "either --cfg or --weights must be specified"
        if args.evolve:
            if args.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                args.project = str(ROOT / "runs/evolve")
            args.exist_ok, args.resume = args.resume, False  # pass resume to exist_ok and disable resume
        args.save_dir = str(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))

    # DDP mode
    device = select_device(args.device, batch_size=args.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not args.image_weights, f"--image-weights {msg}"
        assert not args.evolve, f"--evolve {msg}"
        assert args.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert args.batch_size % WORLD_SIZE == 0, f"--batch-size {args.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not args.evolve:
        train(args.hyp, args, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info("Destroying process group... ")
            dist.destroy_process_group()

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
            "copy_paste"     : (1, 0.0, 1.0)      # segment copy-paste (probability)
        }  

        with open(args.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        args.noval, args.nosave, save_dir = True, True, Path(args.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if args.bucket:
            os.system(f"gsutil cp gs://{args.bucket}/evolve.csv {evolve_csv}")  # download evolve.csv if exists

        for _ in range(args.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: "single" or "weighted"
                x      = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)
                n      = min(5, len(x))  # number of previous results to consider
                x      = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w      = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr   = np.random
                npr.seed(int(time.time()))
                g  = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v  = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)   # significant digits

            # Train mutation
            results   = train(hyp.copy(), args, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, args.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f"Hyperparameter evolution finished {args.evolve} generations\n"
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f"Usage example: $ python train.py --hyp {evolve_yaml}")


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    args = parse_args(True)
    for k, v in kwargs.items():
        setattr(args, k, v)
    main(args)
    return args


if __name__ == "__main__":
    main(parse_args())
