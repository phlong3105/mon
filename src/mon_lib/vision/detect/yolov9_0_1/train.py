#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from mon import core, nn
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    check_amp, check_dataset, check_file, check_img_size, check_suffix, check_yaml, colorstr, get_latest_run,
    increment_path, init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, LOGGER, methods,
    one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer, TQDM_BAR_FORMAT, yaml_save,
)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss_tal_dual import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    de_parallel, EarlyStopping, ModelEMA, select_device, smart_DDP, smart_optimizer, smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK    = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK          = int(os.getenv("RANK",       -1))
WORLD_SIZE    = int(os.getenv("WORLD_SIZE",  1))
GIT_INFO      = None

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    cfg        = opt.model
    weights    = opt.weights
    weights    = weights[0] if isinstance(weights, list | tuple) else weights
    data       = opt.data
    save_dir   = core.Path(opt.save_dir)
    epochs     = opt.epochs
    batch_size = opt.batch_size
    single_cls = opt.single_cls
    evolve     = opt.evolve
    resume     = opt.resume
    noval      = opt.noval
    nosave     = opt.nosave
    workers    = opt.workers
    freeze     = opt.freeze
    
    callbacks.run("on_pretrain_routine_start")

    # Directories
    w    = save_dir / "weights"           
    w.mkdir(parents=True, exist_ok=True)
    last = w / "last.pt"
    best = w / "best.pt"
    
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    hyp["anchor_t"] = 5.0
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
             epochs, hyp, batch_size = opt.epochs, opt.hyp, opt.batch_size
            # weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda  = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
        
    train_path = data_dict["train"]
    val_path   = data_dict["val"]
    nc         = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names      = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    # is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')  # COCO dataset

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
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        # v.requires_grad = True  # train all layers TODO: uncomment this line as in master
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs    = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs        = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer  = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp["lrf"]
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp["lrf"], epochs)  # flat cosine 1->hyp["lrf"]        
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.info("WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.")
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
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
        cache         = None if opt.cache == "val" else opt.cache,
        rect          = opt.rect,
        rank          = LOCAL_RANK,
        workers       = workers,
        image_weights = opt.image_weights,
        close_mosaic  = opt.close_mosaic != 0,
        quad          = opt.quad,
        prefix        = colorstr("train: "),
        shuffle       = True,
        min_items     = opt.min_items,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc    = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            path       = val_path,
            imgsz      = imgsz,
            batch_size = batch_size // WORLD_SIZE * 2,
            stride     = gs,
            single_cls = single_cls,
            hyp        = hyp,
            cache      = None if noval else opt.cache,
            rect       = True,
            rank       = -1,
            workers    = workers * 2,
            pad        = 0.5,
            prefix     = colorstr("val: "),
        )[0]

        if not resume:
            # if not opt.noautoanchor:
            #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # hyp['box'] *= 3 / nl  # scale to layers
    # hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc    = nc  # attach number of classes to model
    model.hyp   = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0            = time.time()
    nb            = len(train_loader)  # number of batches
    nw            = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps          = np.zeros(nc)  # mAP per class
    results       = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler        = torch.cuda.amp.GradScaler(enabled=amp)
    stopper       = EarlyStopping(patience=opt.patience)
    stop          = False
    compute_loss  = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        if epoch == (epochs - opt.close_mosaic):
            LOGGER.info("Closing dataloader mosaic")
            dataset.mosaic = False

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5) % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------
        
        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    opt          = opt,
                    data         = data_dict,
                    batch_size   = batch_size   // WORLD_SIZE * 2,
                    imgsz        = imgsz,
                    conf_thres   = opt.conf_thres,
                    iou_thres    = opt.iou_thres,
                    max_det      = opt.max_det,
                    half         = amp,
                    model        = ema.ema,
                    single_cls   = single_cls,
                    dataloader   = val_loader,
                    save_dir     = save_dir,
                    plots        = False,
                    callbacks    = callbacks,
                    compute_loss = compute_loss
                )

            # Update best mAP
            fi   = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch"       : epoch,
                    "best_fitness": best_fitness,
                    "model"       : deepcopy(de_parallel(model)).half(),
                    "ema"         : deepcopy(ema.ema).half(),
                    "updates"     : ema.updates,
                    "optimizer"   : optimizer.state_dict(),
                    "opt"         : vars(opt),
                    "git"         : GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date"        : datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        opt          = opt,
                        data         = data_dict,
                        batch_size   = batch_size // WORLD_SIZE * 2,
                        imgsz        = imgsz,
                        conf_thres   = opt.conf_thres,
                        iou_thres    = opt.iou_thres,
                        max_det      = opt.max_det,
                        model        = attempt_load(f, device).half(),
                        single_cls   = single_cls,
                        dataloader   = val_loader,
                        save_dir     = save_dir,
                        save_json    = is_coco,
                        verbose      = True,
                        plots        = plots,
                        callbacks    = callbacks,
                        compute_loss = compute_loss
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results

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
@click.option("--local-rank", type=int, default=-1,   help="DDP parameter, do not modify.")
@click.option("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
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
    local_rank: int,
    device    : str,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root      or args["root"]
    root     = core.Path(root)
    weights  = weights   or args["weights"]
    model    = core.Path(model or args["model"])
    model    = model if model.exists() else _current_dir / "config"  / model.name
    model    = model.config_file()
    data     = core.Path(args["data"])
    data     = data  if data.exists() else _current_dir / "data" / data.name
    data     = data.config_file()
    project  = root.name or args["project"]
    fullname = fullname  or args["name"]
    save_dir = save_dir  or root / "run" / "train" / fullname
    save_dir = core.Path(save_dir)
    device   = device    or args["device"]
    hyp      = core.Path(args["hyp"])
    hyp      = hyp if hyp.exists() else _current_dir / "data/hyps" / hyp.name
    hyp      = hyp.yaml_file()
    epochs   = epochs    or args["epochs"]
    exist_ok = exist_ok  or args["exist_ok"]
    verbose  = verbose   or args["verbose"]
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = str(model)
    args["data"]       = str(data)
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
    args["device"]     = device
    args["local_rank"] = local_rank
    args["hyp"]        = str(hyp)
    args["epochs"]     = epochs
    args["steps"]      = steps
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    
    opt = argparse.Namespace(**args)
    
    if not exist_ok:
        core.delete_dir(paths=core.Path(opt.save_dir))
    core.Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()
        # check_requirements()

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last     = core.Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.model, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data    = check_file(opt.data)
        opt.model   = check_yaml(opt.model)
        opt.hyp     = check_yaml(opt.hyp)
        opt.weights = str(opt.weights)
        opt.project = str(opt.project) 
        assert len(opt.model) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            opt.save_dir = root / "run" / "evolve" / project / fullname
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = core.Path(opt.model).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(core.Path(opt.save_dir), exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLO Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks=Callbacks())

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0"            : (1  , 1e-5, 1e-1) ,  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf"            : (1  , 0.01, 1.0)  ,  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum"       : (0.3, 0.6 , 0.98) ,  # SGD momentum/Adam beta1
            "weight_decay"   : (1  , 0.0 , 0.001),  # optimizer weight decay
            "warmup_epochs"  : (1  , 0.0 , 5.0)  ,  # warmup epochs (fractions ok)
            "warmup_momentum": (1  , 0.0 , 0.95) ,  # warmup initial momentum
            "warmup_bias_lr" : (1  , 0.0 , 0.2)  ,  # warmup initial bias lr
            "box"            : (1  , 0.02, 0.2)  ,  # box loss gain
            "cls"            : (1  , 0.2 , 4.0)  ,  # cls loss gain
            "cls_pw"         : (1  , 0.5 , 2.0)  ,  # cls BCELoss positive_weight
            "obj"            : (1  , 0.2 , 4.0)  ,  # obj loss gain (scale with pixels)
            "obj_pw"         : (1  , 0.5 , 2.0)  ,  # obj BCELoss positive_weight
            "iou_t"          : (0  , 0.1 , 0.7)  ,  # IoU training threshold
            "anchor_t"       : (1  , 2.0 , 8.0)  ,  # anchor-multiple threshold
            "anchors"        : (2  , 2.0 , 10.0) ,  # anchors per output grid (0 to ignore)
            "fl_gamma"       : (0  , 0.0 , 2.0)  ,  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h"          : (1  , 0.0 , 0.1)  ,  # image HSV-Hue augmentation (fraction)
            "hsv_s"          : (1  , 0.0 , 0.9)  ,  # image HSV-Saturation augmentation (fraction)
            "hsv_v"          : (1  , 0.0 , 0.9)  ,  # image HSV-Value augmentation (fraction)
            "degrees"        : (1  , 0.0 , 45.0) ,  # image rotation (+/- deg)
            "translate"      : (1  , 0.0 , 0.9)  ,  # image translation (+/- fraction)
            "scale"          : (1  , 0.0 , 0.9)  ,  # image scale (+/- gain)
            "shear"          : (1  , 0.0 , 10.0) ,  # image shear (+/- deg)
            "perspective"    : (0  , 0.0 , 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud"         : (1  , 0.0 , 1.0)  ,  # image flip up-down (probability)
            "fliplr"         : (0  , 0.0 , 1.0)  ,  # image flip left-right (probability)
            "mosaic"         : (1  , 0.0 , 1.0)  ,  # image mixup (probability)
            "mixup"          : (1  , 0.0 , 1.0)  ,  # image mixup (probability)
            "copy_paste"     : (1  , 0.0 , 1.0)} ,  # segment copy-paste (probability)

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, core.Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            os.system(f"gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}")  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
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
                g     = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng    = len(meta)
                v     = np.ones(ng)
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
            results   = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = (
                "metrics/precision", "metrics/recall", "metrics/mAP@0.5", "metrics/mAP@[0.5:0.95]",
                "val/box_loss", "val/obj_loss", "val/cls_loss"
            )
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )
        
        return str(opt.save_dir)
        

if __name__ == "__main__":
    main()

# endregion
