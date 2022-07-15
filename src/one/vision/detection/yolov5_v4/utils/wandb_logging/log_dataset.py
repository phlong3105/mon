#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse

import yaml

from wandb_utils import WandbLogger

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # aicity2021 dict
    logger = WandbLogger(opt, '', None, data, job_type='Dataset Creation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aicity2021', type=str, default='aicity2021/coco128.yaml', help='aicity2021.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    opt = parser.parse_args()

    create_dataset_artifact(opt)
