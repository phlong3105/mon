#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DEIM model training pipeline for object detection.

References:
    - Paper: "DEIM: DETR with Improved Matching for Fast convergence," CVPR 2025.
    - Code: https://github.com/ShihuaHuang95/DEIM
"""

# import os
# import sys
from pprint import pprint

import box
import torch

# noinspection PyUnusedImports
import engine as deim
import mon
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
debug = False

if debug:
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def train(args: dict | box.Box) -> str:
    # Start
    if safe_get_rank() == 0:
        mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
 
    # Pretrained
    if args.weights and args.weights.is_weights_file(exist=True):
        resume = args.weights
        tuning = None
    elif args.resume and args.resume.is_weights_file(exist=True):
        resume = args.resume
        tuning = None
    else:
        resume = None
        tuning = args.tuning
    assert not all([tuning, resume]), "Only support from scratch or resume or tuning at one time."

    # Trainer
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    cfg_path     = root_dir / "option" / args.cfg
    updated_cfg  = args.updated_cfg
    updated_cfg |= {"tuning": str(tuning)} if tuning else {}
    updated_cfg |= {"resume": str(resume)} if resume else {}
    updated_cfg |= {"device": device}      if not args.torchrun else {}
    updated_cfg |= {
        "seed"            : args.seed,
        "output_dir"      : str(args.save_dir),
        "summary_dir"     : str(args.save_dir),
        "test_only"       : args.test_only,
        "print_method"    : args.print_method,
        "print_rank"      : args.print_rank,
        "epochs"          : args.epochs,
        "total_batch_size": args.batch_size,
    }
    cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg)

    if resume or tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    # Train
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    # Finish
    dist_utils.cleanup()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
