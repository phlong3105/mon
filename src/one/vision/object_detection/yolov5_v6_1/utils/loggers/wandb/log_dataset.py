#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse

from onedetection.models.yolov5_v6_1.utils.general import LOGGER

from wandb_utils import WandbLogger

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


# MARK: - Functional

def create_dataset_artifact(opt):
    logger = WandbLogger(opt, None, job_type='Dataset Creation')  # TODO: return value unused
    if not logger.wandb:
        LOGGER.info("install wandb using `pip install wandb` to log the dataset")


# MARK: - Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/coco128.yaml", type=str, help="data.yaml path")
    parser.add_argument("--single-cls", default=False,               action="store_true", help="train as single-class dataset")
    parser.add_argument("--project",    default="YOLOv5",            type=str, help="name of W&B Project")
    parser.add_argument("--entity",     default=None,                help="W&B entity")
    parser.add_argument("--name",       default="log dataset",       type=str, help="name of W&B run")

    opt        = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job

    create_dataset_artifact(opt)
