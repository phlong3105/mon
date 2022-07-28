#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from torch import Tensor

from one.core import AUGMENTS
from one.core import get_image_hw
from one.core import get_num_channels
from one.core import to_tensor
from one.data1.augment.base import BaseAugment
from one.data1.augment.base import BaseAugmentModule
from one.data1.augment.utils import apply_transform_op

__all__ = [
    "ImageAugmentModule",
    "ImageAutoAugment",
    "ImageRandAugment",
    "ImageTrivialAugmentWide",
]


# MARK: - Modules

@AUGMENTS.register(name="image_augment_module")
class ImageAugmentModule(BaseAugmentModule):

    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        """Transform image.
        
        Args:
            input (Tensor[*, C, H, W]):
                Input image.

        Returns:
            input (Tensor[B, C, H, W]):
                Transformed image.
        """
        # NOTE: Transform
        if self.random_transform:
            index      = int(torch.randint(0, self.num_transforms - 1, (1,)))
            transforms = self.transforms[index]
        else:
            transforms = self.transforms
        input = transforms(input)

        # NOTE: Convert to tensor
        if self.to_tensor:
            input = to_tensor(input, normalize=True)

        return input


@AUGMENTS.register(name="image_auto_augment")
class ImageAutoAugment(BaseAugment):
    r"""AutoAugment data augmentation method based on `"AutoAugment: Learning
    Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`.
    If the image is Tensor, it should be of type torch.uint8, and it is
    expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary
    number of leading dimensions. If img is PIL Image, it is expected to be in
    mode "L" or "RGB".

    Args:
        policy (str):
			Augmentation policy. One of: [`imagenet`, `cifar10`, `svhn`].
			Default: `imagenet`.
    """

    cfgs = {
        "imagenet": [
            # (op_name, p, magnitude, num_magnitude_bins=10)
            (("posterize", 0.4, 8   , 10)  , ("rotate"       , 0.6, 9   , 10)),
            (("solarize" , 0.6, 5   , 10)  , ("auto_contrast", 0.6, None, None)),
            (("equalize" , 0.8, None, None), ("equalize"     , 0.6, None, None)),
            (("posterize", 0.6, 7   , 10)  , ("posterize"    , 0.6, 6   , 10)),
            (("equalize" , 0.4, None, None), ("solarize"     , 0.2, 4   , 10)),
            (("equalize" , 0.4, None, None), ("rotate"       , 0.8, 8   , 10)),
            (("solarize" , 0.6, 3   , 10)  , ("equalize"     , 0.6, None, None)),
            (("posterize", 0.8, 5   , 10)  , ("equalize"     , 1.0, None, None)),
            (("rotate"   , 0.2, 3   , 10)  , ("solarize"     , 0.6, 8   , 10)),
            (("equalize" , 0.6, None, None), ("posterize"    , 0.4, 6   , 10)),
            (("rotate"   , 0.8, 8   , 10)  , ("color"        , 0.4, 0   , 10)),
            (("rotate"   , 0.4, 9   , 10)  , ("equalize"     , 0.6, None, None)),
            (("equalize" , 0.0, None, None), ("equalize"     , 0.8, None, None)),
            (("invert"   , 0.6, None, None), ("equalize"     , 1.0, None, None)),
            (("color"    , 0.6, 4   , 10)  , ("contrast"     , 1.0, 8   , 10)),
            (("rotate"   , 0.8, 8   , 10)  , ("color"        , 1.0, 2   , 10)),
            (("color"    , 0.8, 8   , 10)  , ("solarize"     , 0.8, 7   , 10)),
            (("sharpness", 0.4, 7   , 10)  , ("invert"       , 0.6, None, None)),
            (("hshear"   , 0.6, 5   , 10)  , ("equalize"     , 1.0, None, None)),
            (("color"    , 0.4, 0   , 10)  , ("equalize"     , 0.6, None, None)),
            (("equalize" , 0.4, None, None), ("solarize"     , 0.2, 4   , 10)),
            (("solarize" , 0.6, 5   , 10)  , ("auto_contrast", 0.6, None, None)),
            (("invert"   , 0.6, None, None), ("equalize"     , 1.0, None, None)),
            (("color"    , 0.6, 4   , 10)  , ("contrast"     , 1.0, 8   , 10)),
            (("equalize" , 0.8, None, None), ("equalize"     , 0.6, None, None)),
        ],
        "cifar10": [
            # (op_name, p, magnitude, num_magnitude_bins=10)
            (("invert"       , 0.1, None, None), ("contrast"     , 0.2, 6   , 10)),
            (("rotate"       , 0.7, 2   , 10)  , ("htranslate"   , 0.3, 9   , 10)),
            (("sharpness"    , 0.8, 1   , 10)  , ("sharpness"    , 0.9, 3   , 10)),
            (("vshear"       , 0.5, 8   , 10)  , ("vtranslate"   , 0.7, 9   , 10)),
            (("auto_contrast", 0.5, None, None), ("equalize"     , 0.9, None, None)),
            (("vshear"       , 0.2, 7   , 10)  , ("posterize"    , 0.3, 7   , 10)),
            (("color"        , 0.4, 3   , 10)  , ("brightness"   , 0.6, 7   , 10)),
            (("sharpness"    , 0.3, 9   , 10)  , ("brightness"   , 0.7, 9   , 10)),
            (("equalize"     , 0.6, None, None), ("equalize"     , 0.5, None, None)),
            (("contrast"     , 0.6, 7   , 10)  , ("sharpness"    , 0.6, 5   , 10)),
            (("color"        , 0.7, 7   , 10)  , ("htranslate"   , 0.5, 8   , 10)),
            (("equalize"     , 0.3, None, None), ("auto_contrast", 0.4, None, None)),
            (("vtranslate"   , 0.4, 3   , 10)  , ("sharpness"    , 0.2, 6   , 10)),
            (("brightness"   , 0.9, 6   , 10)  , ("color"        , 0.2, 8   , 10)),
            (("solarize"     , 0.5, 2   , 10)  , ("invert"       , 0.0, None, None)),
            (("equalize"     , 0.2, None, None), ("auto_contrast", 0.6, None, None)),
            (("equalize"     , 0.2, None, None), ("equalize"     , 0.6, None, None)),
            (("color"        , 0.9, 9   , 10)  , ("equalize"     , 0.6, None, None)),
            (("auto_contrast", 0.8, None, None), ("solarize"     , 0.2, 8   , 10)),
            (("brightness"   , 0.1, 3   , 10)  , ("color"        , 0.7, 0   , 10)),
            (("solarize"     , 0.4, 5   , 10)  , ("auto_contrast", 0.9, None, None)),
            (("vtranslate"   , 0.9, 9   , 10)  , ("vtranslate"   , 0.7, 9   , 10)),
            (("auto_contrast", 0.9, None, None), ("solarize"     , 0.8, 3   , 10)),
            (("equalize"     , 0.8, None, None), ("invert"       , 0.1, None, None)),
            (("vtranslate"   , 0.7, 9   , 10)  , ("auto_contrast", 0.9, None, None)),
        ],
        "svhn": [
            # (op_name, p, magnitude, num_magnitude_bins=10)
            (("hshear"  , 0.9, 4   , 10)  , ("invert"       , 0.2, None, None)),
            (("vshear"  , 0.9, 8   , 10)  , ("invert"       , 0.7, None, None)),
            (("equalize", 0.6, None, None), ("solarize"     , 0.6, 6   , 10)),
            (("invert"  , 0.9, None, None), ("equalize"     , 0.6, None, None)),
            (("equalize", 0.6, None, None), ("rotate"       , 0.9, 3   , 10)),
            (("hshear"  , 0.9, 4   , 10)  , ("auto_contrast", 0.8, None, None)),
            (("vshear"  , 0.9, 8   , 10)  , ("invert"       , 0.4, None, None)),
            (("vshear"  , 0.9, 5   , 10)  , ("solarize"     , 0.2, 6   , 10)),
            (("invert"  , 0.9, None, None), ("auto_contrast", 0.8, None, None)),
            (("equalize", 0.6, None, None), ("rotate"       , 0.9, 3   , 10)),
            (("hshear"  , 0.9, 4   , 10)  , ("solarize"     , 0.3, 3   , 10)),
            (("vshear"  , 0.8, 8   , 10)  , ("invert"       , 0.7, None, None)),
            (("equalize", 0.9, None, None), ("vtranslate"   , 0.6, 6   , 10)),
            (("invert"  , 0.9, None, None), ("equalize"     , 0.6, None, None)),
            (("contrast", 0.3, 3   , 10)  , ("rotate"       , 0.8, 4   , 10)),
            (("invert"  , 0.8, None, None), ("vtranslate"   , 0.0, 2   , 10)),
            (("vshear"  , 0.7, 6   , 10)  , ("solarize"     , 0.4, 8   , 10)),
            (("invert"  , 0.6, None, None), ("rotate"       , 0.8, 4   , 10)),
            (("vshear"  , 0.3, 7   , 10)  , ("htranslate"   , 0.9, 3   , 10)),
            (("hshear"  , 0.1, 6   , 10)  , ("invert"       , 0.6, None, None)),
            (("solarize", 0.7, 2   , 10)  , ("vtranslate"   , 0.6, 7   , 10)),
            (("vshear"  , 0.8, 4   , 10)  , ("invert"       , 0.8, None, None)),
            (("hshear"  , 0.7, 9   , 10)  , ("vtranslate"   , 0.8, 3   , 10)),
            (("vshear"  , 0.8, 5   , 10)  , ("auto_contrast", 0.7, None, None)),
            (("hshear"  , 0.7, 2   , 10)  , ("invert"       , 0.1, None, None)),
        ],
    }
    
    # MARK: Magic Functions

    def __init__(self, policy: str = "imagenet", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if policy not in self.cfgs:
            raise ValueError(f"`policy` must be one of: {self.cfgs.keys()}."
                             f" But got: {policy}")
        self.transforms = self.cfgs[policy]

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f"(policy={self.policy}, fill={self.fill})"

    # MARK: Configure

    def _augmentation_space(
        self, num_bins: int, image_size: list[int]
    ) -> dict[str, tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "auto_contrast": (torch.tensor(0.0), False),
            "brightness"   : (torch.linspace(0.0, 0.9,   num_bins), True),
            "color"        : (torch.linspace(0.0, 0.9,   num_bins), True),
            "contrast"     : (torch.linspace(0.0, 0.9,   num_bins), True),
            "equalize"     : (torch.tensor(0.0), False),
            "hflip"        : (torch.tensor(0.0), False),
            "vflip"        : (torch.tensor(0.0), False),
            "identity"     : (torch.tensor(0.0), False),
            "invert"       : (torch.tensor(0.0), False),
            "posterize"    : (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "rotate"       : (torch.linspace(0.0, 30.0,  num_bins), True),
            "sharpness"    : (torch.linspace(0.0, 0.9,   num_bins), True),
            "hshear"       : (torch.linspace(0.0, 0.3,   num_bins), True),
            "vshear"       : (torch.linspace(0.0, 0.3,   num_bins), True),
            "solarize"     : (torch.linspace(255.0, 0.0, num_bins), False),
            "htranslate"   : (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "vtranslate"   : (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        }

    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        """Transform image.
        
        Args:
            input (Tensor[*, C, H, W]):
                Image to be transformed.

        Returns:
            input (Tensor[B, C, H, W]):
                Transformed image.
        """
        # NOTE: Fill
        fill = self.fill
        if isinstance(input, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * get_num_channels(input)
            elif fill is not None:
                fill = [float(f) for f in fill]
        
        # NOTE: Transform
        transform_id = int(torch.randint(len(self.transforms), (1,)).item())
        num_ops      = len(self.transforms[transform_id])
        probs        = torch.rand((num_ops,))
        signs        = torch.randint(2, (num_ops,))
        for i, (op_name, p, magnitude, num_magnitude_bins) in enumerate(
            self.transforms[transform_id]
        ):
            if probs[i] <= p:
                op_meta = self._augmentation_space(
                    num_magnitude_bins if num_magnitude_bins is not None else 10,
                    get_image_hw(input)
                )
                magnitudes, signed = op_meta[op_name]
                magnitude = (float(magnitudes[magnitude].item())
                             if magnitude is not None else 0.0)
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                input = apply_transform_op(
                    input         = input,
                    op_name       = op_name,
                    magnitude     = magnitude,
                    interpolation = self.interpolation,
                    fill          = fill
                )
        
        # NOTE: Convert to tensor
        if self.to_tensor:
            input = to_tensor(input, normalize=True)
   
        return input


@AUGMENTS.register(name="image_rand_augment")
class ImageRandAugment(BaseAugment):
    r"""RandAugment data augmentation method based on `"RandAugment: Practical
    automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`.
    
    If the image is Tensor, it should be of type torch.uint8, and it is
    expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary
    number of leading dimensions. If img is PIL Image, it is expected to be in
    mode "L" or "RGB".

    Args:
        num_ops (int):
            Number of augmentation transformations to apply sequentially.
        magnitude (int):
            Magnitude for all the transformations.
        num_magnitude_bins (int):
            Number of different magnitude values.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_ops           : int = 2,
        magnitude         : int = 9,
        num_magnitude_bins: int = 31,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_ops            = num_ops
        self.magnitude          = magnitude
        self.num_magnitude_bins = num_magnitude_bins
    
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_ops={num_ops}, "
        s += "magnitude={magnitude}, "
        s += "num_magnitude_bins={num_magnitude_bins}, "
        s += "interpolation={interpolation}, "
        s += "fill={fill}"
        s += ")"
        return s.format(**self.__dict__)
    
    # MARK: Configure
    
    def _augmentation_space(
        self, num_bins: int, image_size: list[int]
    ) -> dict[str, tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "auto_contrast": (torch.tensor(0.0), False),
            "brightness"   : (torch.linspace(0.0, 0.9,   num_bins), True),
            "color"        : (torch.linspace(0.0, 0.9,   num_bins), True),
            "contrast"     : (torch.linspace(0.0, 0.9,   num_bins), True),
            "equalize"     : (torch.tensor(0.0), False),
            "hflip"        : (torch.tensor(0.0), False),
            "vflip"        : (torch.tensor(0.0), False),
            "identity"     : (torch.tensor(0.0), False),
            "invert"       : (torch.tensor(0.0), False),
            "posterize"    : (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "rotate"       : (torch.linspace(0.0, 30.0,  num_bins), True),
            "sharpness"    : (torch.linspace(0.0, 0.9,   num_bins), True),
            "hshear"       : (torch.linspace(0.0, 0.3,   num_bins), True),
            "vshear"       : (torch.linspace(0.0, 0.3,   num_bins), True),
            "solarize"     : (torch.linspace(255.0, 0.0, num_bins), False),
            "htranslate"   : (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "vtranslate"   : (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        }
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        """Transform image.
        
        Args:
            input (Tensor[*, C, H, W]):
                Image to be transformed.

        Returns:
            input (Tensor[B, C, H, W]):
                Transformed image.
        """
        # NOTE: Fill
        fill = self.fill
        if isinstance(input, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * get_num_channels(input)
            elif fill is not None:
                fill = [float(f) for f in fill]
        
        # NOTE: Transform
        for _ in range(self.num_ops):
            op_meta   = self._augmentation_space(
                self.num_magnitude_bins, get_image_hw(input)
            )
            op_index  = int(torch.randint(len(op_meta), (1,)).item())
            op_name   = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (float(magnitudes[self.magnitude].item())
                         if magnitudes.ndim > 0 else 0.0)
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            input = apply_transform_op(
                input         = input,
                op_name       = op_name,
                magnitude     = magnitude,
                interpolation = self.interpolation,
                fill          = fill
            )
        
        # NOTE: Convert to tensor
        if self.to_tensor:
            input = to_tensor(input, normalize=True)

        return input


@AUGMENTS.register(name="image_trivial_augment_wide")
class ImageTrivialAugmentWide(BaseAugment):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as
    described in `"TrivialAugment: Tuning-free Yet State-of-the-Art Data
    Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is Tensor, it should be of type torch.uint8, and it is
    expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary
    number of leading dimensions. If img is PIL Image, it is expected to be in
    mode "L" or "RGB".

    Args:
        num_magnitude_bins (int):
            Number of different magnitude values.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, num_magnitude_bins: int = 31, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_magnitude_bins = num_magnitude_bins
    
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_magnitude_bins={num_magnitude_bins}, "
        s += "interpolation={interpolation}, "
        s += "fill={fill}"
        s += ")"
        return s.format(**self.__dict__)

    # MARK: Configure
    
    def _augmentation_space(self, num_bins: int) -> dict[str, tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "auto_contrast": (torch.tensor(0.0), False),
            "brightness"   : (torch.linspace(0.0, 0.99,  num_bins), True),
            "color"        : (torch.linspace(0.0, 0.99,  num_bins), True),
            "contrast"     : (torch.linspace(0.0, 0.99,  num_bins), True),
            "equalize"     : (torch.tensor(0.0), False),
            "hflip"        : (torch.tensor(0.0), False),
            "vflip"        : (torch.tensor(0.0), False),
            "identity"     : (torch.tensor(0.0), False),
            "invert"       : (torch.tensor(0.0), False),
            "posterize"    : (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "rotate"       : (torch.linspace(0.0, 135.0, num_bins), True),
            "sharpness"    : (torch.linspace(0.0, 0.99,  num_bins), True),
            "hshear"       : (torch.linspace(0.0, 0.99,  num_bins), True),
            "vshear"       : (torch.linspace(0.0, 0.99,  num_bins), True),
            "solarize"     : (torch.linspace(255.0, 0.0, num_bins), False),
            "htranslate"   : (torch.linspace(0.0, 32.0,  num_bins), True),
            "vtranslate"   : (torch.linspace(0.0, 32.0,  num_bins), True),
        }
    
    # MARK: Forward Pass
    
    def forward(self, input: Tensor) -> Tensor:
        """Transform image.
        
        Args:
            input (Tensor[*, C, H, W]):
                Image to be transformed.

        Returns:
            input (Tensor[B, C, H, W]):
                Transformed image.
        """
        # NOTE: Fill
        fill = self.fill
        if isinstance(input, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * get_num_channels(input)
            elif fill is not None:
                fill = [float(f) for f in fill]
        
        # NOTE: Transform
        op_meta   = self._augmentation_space(self.num_magnitude_bins)
        op_index  = int(torch.randint(len(op_meta), (1,)).item())
        op_name   = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                     if magnitudes.ndim > 0 else 0.0)
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        
        input = apply_transform_op(
            input         = input,
            op_name       = op_name,
            magnitude     = magnitude,
            interpolation = self.interpolation,
            fill          = fill
        )
        
        # NOTE: Convert to tensor
        if self.to_tensor:
            input = to_tensor(input, normalize=True)

        return input
