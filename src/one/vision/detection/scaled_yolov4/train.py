#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from munch import Munch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import \
    one.vision.detection.scaled_yolov4.test as test  # import test.py to get
# mAP after each epoch
from one.constants import DATA_DIR
from one.constants import PRETRAINED_DIR
from one.vision.detection.scaled_yolov4.models.yolo import Model
from one.vision.detection.scaled_yolov4.utils.datasets import create_dataloader
from one.vision.detection.scaled_yolov4.utils.general import check_anchors
from one.vision.detection.scaled_yolov4.utils.general import check_file
from one.vision.detection.scaled_yolov4.utils.general import check_git_status
from one.vision.detection.scaled_yolov4.utils.general import check_img_size
from one.vision.detection.scaled_yolov4.utils.general import compute_loss
from one.vision.detection.scaled_yolov4.utils.general import fitness
from one.vision.detection.scaled_yolov4.utils.general import get_latest_run
from one.vision.detection.scaled_yolov4.utils.general import increment_dir
from one.vision.detection.scaled_yolov4.utils.general import \
    labels_to_class_weights
from one.vision.detection.scaled_yolov4.utils.general import \
    labels_to_image_weights
from one.vision.detection.scaled_yolov4.utils.general import plot_evolution
from one.vision.detection.scaled_yolov4.utils.general import plot_images
from one.vision.detection.scaled_yolov4.utils.general import plot_labels
from one.vision.detection.scaled_yolov4.utils.general import plot_results
from one.vision.detection.scaled_yolov4.utils.general import print_mutation
from one.vision.detection.scaled_yolov4.utils.general import strip_optimizer
from one.vision.detection.scaled_yolov4.utils.general import \
    torch_distributed_zero_first
from one.vision.detection.scaled_yolov4.utils.google_utils import \
    attempt_download
from one.vision.detection.scaled_yolov4.utils.torch_utils import init_seeds
from one.vision.detection.scaled_yolov4.utils.torch_utils import intersect_dicts
from one.vision.detection.scaled_yolov4.utils.torch_utils import ModelEMA
from one.vision.detection.scaled_yolov4.utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Scaled-YOLOv4 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# H1: - Functional -------------------------------------------------------------

def train(
    hyp,
    args: dict | Munch | argparse.Namespace,
    device,
    tb_writer=None
):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
    
    print(f"Hyperparameters {hyp}")
    print(tb_writer.log_dir)
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(args.logdir) / "evolve"  # logging directory
    wdir    = str(log_dir / "weights") + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last         = wdir + "last.pt"
    best         = wdir + "best.pt"
    results_file = str(log_dir / "results.txt")
    epochs, batch_size, total_batch_size, weights, rank = \
        args.epochs, args.batch_size, args.total_batch_size, args.weights, args.global_rank

    # TODO: Use DDP logging. Only the first process is allowed to log.
    # Save run settings
    with open(log_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / "args.yaml", "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Configure
    cuda = device.type != "cpu"
    init_seeds(2 + rank)
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    if os.path.isdir(DATA_DIR):
        train_path = os.path.join(DATA_DIR, data_dict["train"])
        test_path  = os.path.join(DATA_DIR, data_dict["val"])
    else:
        train_path = os.path.join(data_dict["path"], data_dict["train"])
        test_path  = os.path.join(data_dict["path"], data_dict["val"])
    nc, names  = (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (len(names), nc, args.data)  # check

    # Model
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt       = torch.load(weights, map_location=device)  # load checkpoint
        model      = Model(args.cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)  # create
        exclude    = ["anchor"] if args.cfg else []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        print("Transferred %g/%g items from %s" % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(args.cfg, ch=3, nc=nc).to(device)# create
        #model = model.to(memory_format=torch.channels_last)  # create

    # Optimizer
    nbs                  = 64  # nominal batch size
    accumulate           = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if ".bias" in k:
            pg2.append(v)  # biases
        elif ".weight" in k and ".bn" not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if args.adam:
        optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    print("Optimizer groups: %g .bias, %g conv.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf        = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # Results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if not args.resume:
            start_epoch = 0
        if epochs < start_epoch:
            print("%s has been trained for %g epochs. Fine-tuning for %g additional epochs." %
                  (weights, ckpt["epoch"], epochs))
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict
    
    # Image sizes
    gs                = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if args.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print("Using SyncBatchNorm()")

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=(args.local_rank))

    # Trainloader
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        args,
        hyp        = hyp,
        augment    = True,
        cache      = args.cache_images,
        rect       = args.rect,
        local_rank = rank,
        world_size = args.world_size
    )
    
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb  = len(dataloader)  # number of batches
    assert mlc < nc, "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (mlc, nc, args.data, nc - 1)

    # Testloader
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = create_dataloader(
            test_path, 
            imgsz_test, 
            batch_size, 
            gs,
            args,
            hyp        = hyp,
            augment    = False,
            cache      = args.cache_images,
            rect       = True,
            local_rank = -1,
            world_size = args.world_size
        )[0]

    # Model parameters
    hyp["cls"]          *= nc / 80.0  # scale coco-tuned hyp["cls"] to current dataset
    model.nc             = nc  # attach number of classes to model
    model.hyp            = hyp  # attach hyperparameters to model
    model.gr             = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights  = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names          = names

    # Class frequency
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c      = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            tb_writer.add_histogram("classes", c, 0)

        # Check anchors
        if not args.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)

    # Start training
    t0                   = time.time()
    nw                   = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw                   = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps                 = np.zeros(nc)  # mAP per class
    results              = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler               = amp.GradScaler(enabled=cuda)
    if rank in [0, -1]:
        print("Image sizes %g train, %g test" % (imgsz, imgsz_test))
        print("Using %g dataloader workers" % dataloader.num_workers)
        print("Starting training for %g epochs..." % epochs)
        
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                w               = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights   = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(
                    range(dataset.n), weights=image_weights,  k=dataset.n
                )  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
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
            print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "GIoU", "obj", "cls", "total", "targets", "img_size"))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni   = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, hyp["momentum"]])

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns   = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward                
                pred = model(imgs)
                #pred = model(imgs.to(memory_format=torch.channels_last))

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                if rank != -1:
                    loss *= args.world_size  # gradient averaged between devices in DDP mode
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem   = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s     = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f      = str(log_dir / ("train_batch%g.jpg" % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats="HWC", global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model, include=["yaml", "nc", "hyp", "gr", "names", "stride"])
            final_epoch = epoch + 1 == epochs
            if not args.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(
                    args.data,
                    batch_size = batch_size,
                    imgsz      = imgsz_test,
                    save_json  = final_epoch and args.data.endswith(os.sep + "coco.yaml"),
                    model      = ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                    single_cls = args.single_cls,
                    dataloader = testloader,
                    save_dir   = log_dir
                )

            # Write
            with open(results_file, "a") as f:
                f.write(s + "%10.4g" * 7 % results + "\n")  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(args.name) and args.bucket:
                os.system("gsutil cp %s gs://%s/results/results%s.txt" % (results_file, args.bucket, args.name))

            # Tensorboard
            if tb_writer:
                tags = [
                    "train/giou_loss", "train/obj_loss", "train/cls_loss",
                    "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
                    "val/giou_loss", "val/obj_loss", "val/cls_loss"
                ]
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not args.nosave) or (final_epoch and not args.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {
                        "epoch"           : epoch,
                        "best_fitness"    : best_fitness,
                        "training_results": f.read(),
                        "model"           : ema.ema.module if hasattr(ema, "module") else ema.ema,
                        "optimizer"       : None if final_epoch else optimizer.state_dict()
                    }

                # Save last, best and delete
                torch.save(ckpt, last)
                if epoch >= (epochs-30):
                    torch.save(ckpt, last.replace(".pt", "_{:03d}.pt".format(epoch)))
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n                      = ("_" if len(args.name) and not args.name.isnumeric() else "") + args.name
        fresults, flast, fbest = "results%s.txt" % n, wdir + "last%s.pt" % n, wdir + "best%s.pt" % n
        for f1, f2 in zip([wdir + "last.pt", wdir + "best.pt", "results.txt"], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith(".pt")  # is *.pt
                strip_optimizer(f2, f2.replace(".pt","_strip.pt")) if ispt else None  # strip optimizer
                os.system("gsutil cp %s gs://%s/weights" % (f2, args.bucket)) if args.bucket and ispt else None  # upload
        # Finish
        if not args.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        print("%g epochs completed in %.3f hours.\n" % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


# H1: - Main -------------------------------------------------------------------

def parse_args():
    weights_dir = PRETRAINED_DIR / "scaled_yolov4"
    parser      = argparse.ArgumentParser()
    parser.add_argument("--weights"     , default=weights_dir/"yolov4_p7_coco.pt", type=str                   , help="initial weights path")
    parser.add_argument("--cfg"         , default=""                             , type=str                   , help="model.yaml path")
    parser.add_argument("--data"        , default=ROOT/"data/coco128.yaml"       , type=str                   , help="data.yaml path")
    parser.add_argument("--hyp"         , default=""                             , type=str                   , help="hyperparameters path, i.e. data/hyp.scratch.yaml")
    parser.add_argument("--epochs"      , default=300                            , type=int)
    parser.add_argument("--batch-size"  , default=16                             , type=int                   , help="total batch size for all GPUs")
    parser.add_argument("--img-size"    , default=[640, 640]                     , nargs="+", type=int        , help="train,test sizes")
    parser.add_argument("--rect"        , default=False                          , action="store_true"        , help="rectangular training")
    parser.add_argument("--resume"      , default=False                          , nargs="?", const="get_last", help="resume from given path/last.pt, or most recent run if blank")
    parser.add_argument("--nosave"      , default=False                          , action="store_true"        , help="only save final checkpoint")
    parser.add_argument("--notest"      , default=False                          , action="store_true"        , help="only test final epoch")
    parser.add_argument("--noautoanchor", default=False                          , action="store_true"        , help="disable autoanchor check")
    parser.add_argument("--evolve"      , default=False                          , action="store_true"        , help="evolve hyperparameters")
    parser.add_argument("--bucket"      , default=""                             , type=str                   , help="gsutil bucket")
    parser.add_argument("--cache-images", default=False                          , action="store_true"        , help="cache images for faster training")
    parser.add_argument("--name"        , default=""                                                          , help="renames results.txt to results_name.txt if supplied")
    parser.add_argument("--device"      , default=""                                                          , help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale" , default=False                          , action="store_true"        , help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls"  , default=False                          , action="store_true"        , help="train as single-class dataset")
    parser.add_argument("--adam"        , default=False                          , action="store_true"        , help="use torch.optim.Adam() optimizer")
    parser.add_argument("--sync-bn"     , default=False                          , action="store_true"        , help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--local_rank"  , default=-1                             , type=int                   , help="DDP parameter, do not modify")
    parser.add_argument("--logdir"      , default="runs/"                        , type=str                   , help="logging directory")
    parser.add_argument("--verbose"     , default="False"                        , type=bool                  , help="Verbosity")
    args = parser.parse_args()
    return args


def main(args: dict | Munch | argparse.Namespace):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
        
    # Resume
    if args.resume:
        last = get_latest_run() if args.resume == "get_last" else args.resume  # resume from most recent run
        if last and not args.weights:
            print(f"Resuming training from {last}")
        args.weights = last if args.resume and not args.weights else args.weights
    if args.local_rank == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()

    args.hyp = args.hyp or ("data/hyp.finetune.yaml" if args.weights else "data/hyp.scratch.yaml")
    args.data, args.cfg, args.hyp = check_file(args.data), check_file(args.cfg), check_file(args.hyp)  # check files
    assert len(args.cfg) or len(args.weights), "either --cfg or --weights must be specified"

    args.img_size.extend([args.img_size[-1]] * (2 - len(args.img_size)))  # extend to 2 sizes (train, test)
    device               = select_device(args.device, batch_size=args.batch_size)
    args.total_batch_size = args.batch_size
    args.world_size       = 1
    args.global_rank      = -1

    # DDP mode
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")  # distributed backend
        args.world_size  = dist.get_world_size()
        args.global_rank = dist.get_rank()
        assert args.batch_size % args.world_size == 0, "--batch-size must be multiple of CUDA device count"
        args.batch_size  = args.total_batch_size // args.world_size

    print(args)
    
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    if not args.evolve:
        tb_writer = None
        if args.global_rank in [-1, 0]:
            print("Start Tensorboard with 'tensorboard --logdir %s', view at http://localhost:6006/" % args.logdir)
            tb_writer = SummaryWriter(log_dir=increment_dir(Path(args.logdir) / "exp", args.name))  # runs/exp

        train(hyp, args, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0"         : (1, 1e-5, 1e-1),    # initial learning rate (SGD=1E-2, Adam=1E-3)
            "momentum"    : (0.1, 0.6, 0.98),   # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),    # optimizer weight decay
            "giou"        : (1, 0.02, 0.2),     # GIoU loss gain
            "cls"         : (1, 0.2, 4.0),      # cls loss gain
            "cls_pw"      : (1, 0.5, 2.0),      # cls BCELoss positive_weight
            "obj"         : (1, 0.2, 4.0),      # obj loss gain (scale with pixels)
            "obj_pw"      : (1, 0.5, 2.0),      # obj BCELoss positive_weight
            "iou_t"       : (0, 0.1, 0.7),      # IoU training threshold
            "anchor_t"    : (1, 2.0, 8.0),      # anchor-multiple threshold
            "fl_gamma"    : (0, 0.0, 2.0),      # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h"       : (1, 0.0, 0.1),      # image HSV-Hue augmentation (fraction)
            "hsv_s"       : (1, 0.0, 0.9),      # image HSV-Saturation augmentation (fraction)
            "hsv_v"       : (1, 0.0, 0.9),      # image HSV-Value augmentation (fraction)
            "degrees"     : (1, 0.0, 45.0),     # image rotation (+/- deg)
            "translate"   : (1, 0.0, 0.9),      # image translation (+/- fraction)
            "scale"       : (1, 0.0, 0.9),      # image scale (+/- gain)
            "shear"       : (1, 0.0, 10.0),     # image shear (+/- deg)
            "perspective" : (1, 0.0, 0.001),    # image perspective (+/- fraction), range 0-0.001
            "flipud"      : (0, 0.0, 1.0),      # image flip up-down (probability)
            "fliplr"      : (1, 0.0, 1.0),      # image flip left-right (probability)
            "mixup"       : (1, 0.0, 1.0)       # image mixup (probability)
        }  

        assert args.local_rank == -1, "DDP mode not implemented for --evolve"
        args.notest, args.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path("runs/evolve/hyp_evolved.yaml")  # save best result here
        if args.bucket:
            os.system("gsutil cp gs://%s/evolve.txt ." % args.bucket)  # download evolve.txt if exists

        for _ in range(100):  # generations to evolve
            if os.path.exists("evolve.txt"):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = "single"  # parent selection method: "single" or "weighted"
                x      = np.loadtxt("evolve.txt", ndmin=2)
                n      = min(5, len(x))  # number of previous results to consider
                x      = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w      = fitness(x) - fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr   = np.random
                npr.seed(int(time.time()))
                g  = np.array([x[0] for x in meta.values()])  # gains 0-1
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
            results = train(hyp.copy(), args, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, args.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print("Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these "
              "hyperparameters: $ python train.py --hyp %s" % (yaml_file, yaml_file))


if __name__ == "__main__":
    main(parse_args())
