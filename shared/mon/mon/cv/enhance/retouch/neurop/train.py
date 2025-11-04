#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NeurOP model training pipeline for image retouching.

References:
    - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
    - Code: https://github.com/amberwangyili/neurop
"""

import os
import random
from collections import defaultdict

import box
import numpy as np
import torch

import mon
from neurop import build_model, build_train_loader, dict_to_nonedict, parse

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    cfg_path = root_dir / "neurop" / "option" / "train" / args.cfg
    cfgs     = parse(str(cfg_path))
    cfgs     = dict_to_nonedict(cfgs)
    cfgs["network_G"]["init_model"] = mon.rt.parse_weights_file(mon.ROOT_DIR, cfgs.network_G.init_model)
    
    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    device = mon.create_device(args.device)
    
    # Seed
    seed = cfgs["train"]["manual_seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Data I/O
    dataset_opt  = cfgs["datasets"]
    train_loader = build_train_loader(dataset_opt)
    
    # Model
    model = build_model(cfgs)
    
    # Data I/O
    args["train_dataloader"]["dataset"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["dataset"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)

    # Training
    current_step = 0
    total_iters  = cfgs["train"]["niter"]
    total_epochs = int(total_iters / len(train_loader))
    with mon.create_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(total_epochs + 1),
            total       = total_epochs + 1,
            description = f"[bright_yellow]Training"
        ):
            for _, train_data in enumerate(train_loader):
                # print(f"{train_data["LQ_path"]} | {train_data["GT_path"]}")
                current_step += 1
                if current_step > total_iters:
                    break
                model.feed_data(train_data)
                model.optimize_parameters()
            
            # Log
            logs    = model.get_current_log()
            message = "[epoch:{:3d}, iter:{:8,d}, ".format(epoch, current_step)
            for k, v in logs.items():
                v /= len(train_loader)
                message += "{:s}: {:.4e} ".format(k, v)
            model.log_dict = defaultdict(int)
            
            # Save
            model.save("latest", save_dir=args.save_dir)

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
