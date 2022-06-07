#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from torch import Tensor

from one.core import AUGMENTS
from one.core import to_tensor
from one.data.augment.base import BaseAugment
from one.data.augment.utils import apply_transform_op

__all__ = [
    "LowHighResImagesAugment",
]


# MARK: - Modules

@AUGMENTS.register(name="lowhighres_images_augment")
class LowHighResImagesAugment(BaseAugment):
    r"""
    
    Args:
        policy (str):
			Augmentation policy. One of: [`x2`, `x3`, `x4`]. Default: `x4`.
    """

    cfgs = {
        "x2": [
            # (op_name, p, magnitude)
            (("lowhighres_images_random_crop", 1.0, (32, 2)),
             ("hflip",  0.5, None),
             ("vflip",  0.5, None),
             ("rotate", 0.5, 90),),
        ],
        "x3": [
            # (op_name, p, magnitude)
            (("lowhighres_images_random_crop", 1.0, (32, 3)),
             ("hflip",  0.5, None),
             ("vflip",  0.5, None),
             ("rotate", 0.5, 90),),
        ],
        "x4": [
            # (op_name, p, magnitude)
            (("lowhighres_images_random_crop", 1.0, (32, 4)),
             ("hflip",  0.5, None),
             ("vflip",  0.5, None),
             ("rotate", 0.5, 90),),
        ],
    }
    
    # MARK: Magic Functions

    def __init__(self, policy: str = "default", *args, **kwargs):
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
    
    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Transform image and target.
        
        Args:
            input (Tensor[*, C, H, W]):
                Image to be transformed.
            target (Tensor[*, C, H, W]):
                Target to be transformed.
                
        Returns:
            input (Tensor[B, C, H, W]):
                Transformed image.
            target (Tensor[B, C, H, W]):
                Transformed target.
        """
        # NOTE: Transform
        transform_id = int(torch.randint(len(self.transforms), (1,)).item())
        num_ops      = len(self.transforms[transform_id])
        probs        = torch.rand((num_ops,))
        for i, (op_name, p, magnitude) in enumerate(self.transforms[transform_id]):
            if probs[i] > p:
                continue
            magnitude = magnitude if magnitude is not None else 0.0
            if op_name in ["lowhighres_images_random_crop"]:
                input, target = apply_transform_op(
                    input         = input,
                    target        = target,
                    op_name       = op_name,
                    magnitude     = magnitude,
                    interpolation = self.interpolation,
                    fill          = self.fill
                )
            else:
                input = apply_transform_op(
                    input         = input,
                    op_name       = op_name,
                    magnitude     = magnitude,
                    interpolation = self.interpolation,
                    fill          = self.fill
                )
                if target is not None:
                    target = apply_transform_op(
                        input         = target,
                        op_name       = op_name,
                        magnitude     = magnitude,
                        interpolation = self.interpolation,
                        fill          = self.fill
                    )
        # NOTE: Convert to tensor
        if self.to_tensor:
            input  = to_tensor(input,  normalize=True)
            target = to_tensor(target, normalize=True) if (target is not None) else None

        return input, target
