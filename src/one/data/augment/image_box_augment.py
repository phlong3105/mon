#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from one.core import AUGMENTS
from one.data.augment.base import BaseAugment
from one.data.augment.utils import apply_transform_op
from one.data.data_class import ObjectAnnotation

__all__ = [
    "ImageBoxAugment",
]


# MARK: - Modules

@AUGMENTS.register(name="image_box_augment")
class ImageBoxAugment(BaseAugment):
    r"""
    
    Args:
        policy (str):
			Augmentation policy. One of: [`scratch`, `finetune`].
			Default: `scratch`.
    """

    cfgs = {
        "scratch": [
            # (op_name, p, magnitude)
            (("image_box_random_perspective", 0.5, (0.0, 0.5, 0.5, 0.0, 0.0)),
             ("adjust_hsv", 0.5, (0.015, 0.7, 0.4)),
             ("hflip_image_box", 0.5, None),
             ("vflip_image_box", 0.5, None),),
        ],
        "finetune": [
            (("image_box_random_perspective", 0.5, (0.0, 0.5, 0.8, 0.0, 0.0)),
             ("adjust_hsv", 0.5, (0.015, 0.7, 0.4)),
             ("hflip_image_box", 0.5, None),
             ("vflip_image_box", 0.5, None),),
        ],
    }
    
    # MARK: Magic Functions

    def __init__(self, policy: str = "scratch", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if policy not in self.cfgs:
            raise ValueError(f"`policy` must be one of: {self.cfgs.keys()}."
                             f" But got: {policy}")
        self.transforms = self.cfgs[policy]

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f"(policy={self.policy}, fill={self.fill})"
    
    # MARK: Configure

    def _augmentation_space(self, *args, **kwargs) -> dict[str, tuple[Tensor, bool]]:
        pass

    # MARK: Forward Pass
    
    def forward(self, input: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        
        Args:
            input (np.ndarray):
                Image to be transformed.
            target (np.ndarray[*, 4):
                Target to be transformed. Boxes in (x, y, x, y) format.
        """
        # NOTE: Transform
        transform_id = int(torch.randint(len(self.transforms), (1,)).item())
        num_ops      = len(self.transforms[transform_id])
        probs        = torch.rand((num_ops,))
        for i, (op_name, p, magnitude) in enumerate(self.transforms[transform_id]):
            if probs[i] > p:
                continue
            magnitude = magnitude if magnitude is not None else 0.0
            
            if op_name == "image_box_random_perspective":
                """
                target[:, 2:6] = box_cxcywh_norm_to_xyxy(
                    target[:, 2:6], input.shape[0], input.shape[1]
                )
                """
                input, target = apply_transform_op(
                    input         = input,
                    target        = target,
                    op_name       = op_name,
                    magnitude     = magnitude,
                    interpolation = self.interpolation,
                    fill          = self.fill
                )
                nl = len(target)  # Number of labels
                if nl:
                    target = target
                else:
                    target = np.zeros((nl, ObjectAnnotation.box_label_len()))
                """
                target[:, 2:6] = box_xyxy_to_cxcywh_norm(
                    target[:, 2:6], input.shape[0], input.shape[1]
                )
                """
            else:
                input, target = apply_transform_op(
                    input         = input,
                    target        = target,
                    op_name       = op_name,
                    magnitude     = magnitude,
                    interpolation = self.interpolation,
                    fill          = self.fill
                )
            '''
            elif op_name == "adjust_hsv":
                input = adjust_hsv(
                    input,
                    h_factor = magnitude[0],
                    s_factor = magnitude[1],
                    v_factor = magnitude[2],
                )
            elif op_name == "hflip":
                input        = np.fliplr(input)
                target[:, 2] = 1 - target[:, 2]
            elif op_name == "vflip":
                input        = np.flipud(input)
                target[:, 3] = 1 - target[:, 3]
            '''
            
        return input, target
