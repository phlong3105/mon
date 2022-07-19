#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compose multiple transformations.
"""

from __future__ import annotations

import inspect
import sys
from abc import ABC
from abc import abstractmethod
from typing import Union

import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn
from torch import Tensor
from torchvision.utils import _log_api_usage_once

from one.core import assert_number_in_range
from one.core import Callable
from one.core import TRANSFORMS


# MARK: - Module

class Compose:
    """Composes several transforms together. This transform does not support
    torchscript. Please, see the note below.

    Args:
        transforms (list[Union[list, dict]], dict):
            List of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    Notes:
        In order to script the transformations, please use
        `torch.nn.Sequential` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with
        `torch.Tensor`, does not require `lambda` functions or `PIL.Image`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        transforms: Union[list[Union[Callable, dict]], dict],
        *args, **kwargs
    ):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        
        if isinstance(transforms, dict):
            transforms = [v for k, v in transforms.items()]
        if isinstance(transforms, list):
            transforms = [
                TRANSFORMS.build_from_dict(cfg=t)
                if isinstance(t, dict) else t
                for t in transforms
            ]
            if not all(isinstance(t, Transform) for t in transforms):
                raise TypeError(f"All items in `transforms` must be callable.")
            
        self.transforms = transforms

    def __call__(
        self,
        input : Union[Tensor, np.ndarray, PIL.Image],
        target: Union[Tensor, np.ndarray, PIL.Image, None] = None,
        *args, **kwargs
    ) -> tuple[
        Union[Tensor, np.ndarray, PIL.Image],
        Union[Tensor, np.ndarray, PIL.Image, None]
    ]:
        """
        
        Args:
            input (Tensor, np.ndarray, PIL.Image):
                Input.
            target (Tensor, np.ndarray, PIL.Image, None):
                Target. Default: `None`.

        Returns:
            input (Tensor, np.ndarray, PIL.Image):
                Transformed input.
            target (Tensor, np.ndarray, PIL.Image, None):
                Transformed target. Default: `None`.
        """
        for t in self.transforms:
            input, target = t(input, target)
        return input, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ComposeScript(nn.Sequential):
    """Composes several transforms together. This transform support torchscript.
    Please, see the note below.

    Args:
        transforms (list of `Transform` objects):
            List of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    Notes:
        In order to script the transformations, please use
        `torch.nn.Sequential` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with
        `torch.Tensor`, does not require `lambda` functions or `PIL.Image`.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        transforms: Union[list[Union[Callable, dict]], dict],
        *args, **kwargs
    ):
        if isinstance(transforms, dict):
            transforms = [v for k, v in transforms.items()]
        if isinstance(transforms, list):
            transforms = [
                TRANSFORMS.build_from_dict(cfg=t)
                if isinstance(t, dict) else t
                for t in transforms
            ]
            if not all(isinstance(t, Transform) for t in transforms):
                raise TypeError(f"All items in `transforms` must be callable.")
        
        args = transforms + args
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Union[Tensor, np.ndarray, PIL.Image],
        target: Union[Tensor, np.ndarray, PIL.Image, None] = None,
        *args, **kwargs
    ) -> tuple[
        Union[Tensor, np.ndarray, PIL.Image],
        Union[Tensor, np.ndarray, PIL.Image, None]
    ]:
        """
        
        Args:
            input (Tensor, np.ndarray, PIL.Image):
                Input.
            target (Tensor, np.ndarray, PIL.Image, None):
                Target. Default: `None`.

        Returns:
            input (Tensor, np.ndarray, PIL.Image):
                Transformed input.
            target (Tensor, np.ndarray, PIL.Image, None):
                Transformed target. Default: `None`.
        """
        for t in self:
            input, target = t(input, target)
        return input, target
    

class Transform(nn.Module, metaclass=ABC):
    """Transform module.
    
    Args:
        p (float):
            Probability of the image being adjusted. Default: `None`.
    """

    # MARK: Magic Functions
    
    def __init__(self, p: Union[float, None] = None, *args, **kwargs):
        super().__init__()
        assert_number_in_range(p, 0.0, 1.0)
        self.p = p
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __call__(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        if self.p is None or torch.rand(1).item() <= self.p:
            return super.__call__(input, target, *args, **kwargs)
        return input, target
        
    # MARK: Forward Pass
    
    @abstractmethod
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        pass
    

# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
