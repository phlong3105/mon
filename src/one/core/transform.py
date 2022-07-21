#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compose multiple transformations.
"""

from __future__ import annotations

import inspect
import sys
from abc import ABC
from abc import abstractmethod

import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn
from torch import Tensor
from torchvision.utils import _log_api_usage_once

from one.core.factory import TRANSFORMS
from one.core.types import assert_number_in_range
from one.core.types import Dict
from one.core.types import ScalarOrCollectionT


# MARK: - Module

class Transform(nn.Module, metaclass=ABC):
    """
    Transform module.
    
    Args:
        p (float):
            Probability of the image being adjusted. Default: `None` means
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(self, p: float = 1.0, *args, **kwargs):
        super().__init__()
        assert_number_in_range(p, 0.0, 1.0)
        self.p = p
    
    def __repr__(self) -> str:
        """
        The `__repr__` function returns a string representation of the object.
        
        Returns:
            The class name.
        """
        return f"{self.__class__.__name__}()"
    
    def __call__(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        If the probability is greater than a random number, then apply
        transforms, otherwise return the input.
        
        Args:
            input (Tensor): The input tensor to be transformed.
            target (Tensor | None): The target tensor to be transformed.
                Defaults to None.
        
        Returns:
            The input and target tensors.
        """
        if self.p is not None or torch.rand(1).item() <= self.p:
            return super.__call__(input, target, *args, **kwargs)
        return input, target
        
    # MARK: Forward Pass
    
    @abstractmethod
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        The function `forward` takes a tensor `input` and an optional tensor
        `target` and returns a tuple of two tensors, the first being the
        transformed input and the second being the transformed target.
        
        Args:
            input (Tensor):  The input tensor to be transformed.
            target (Tensor | None): The target tensor to be transformed.
                Defaults to None.
        
        Returns:
            The input and target tensors.
        """
        pass


class Compose:
    """
    Composes several transforms together. This transform does not support
    torchscript. Please, see the note below.

    Args:
        transforms (ScalarOrCollectionT[Transform | Dict]):
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
        transforms: ScalarOrCollectionT[Transform | Dict],
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
        input : Tensor | np.ndarray | PIL.Image,
        target: Tensor | np.ndarray | PIL.Image | None = None,
        *args, **kwargs
    ) -> tuple[
        Tensor | np.ndarray | PIL.Image,
        Tensor | np.ndarray | PIL.Image | None
    ]:
        """
        It applies the transforms to the input and target.
        
        Args:
            input (Tensor | np.ndarray | PIL.Image): The input tensor to be
                transformed.
            target (Tensor | np.ndarray | PIL.Image | None): The target tensor
                to be transformed.
        
        Returns:
            The transformed input and target.
        """
        for t in self.transforms:
            input, target = t(input, target)
        return input, target

    def __repr__(self) -> str:
        """
        The function returns a string that contains the name of the class,
        and the string representation of each transform in the list of
        transforms.
        
        Returns:
            A string representation of the object.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ComposeScript(nn.Sequential):
    """
    Composes several transforms together. This transform support torchscript.
    Please, see the note below.

    Args:
        transforms (ScalarOrCollectionT[Transform | Dict]):
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
        transforms: ScalarOrCollectionT[Transform | Dict],
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
        
        args = transforms + list(args)
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        input : Tensor | np.ndarray | PIL.Image,
        target: Tensor | np.ndarray | PIL.Image | None = None,
        *args, **kwargs
    ) -> tuple[
        Tensor | np.ndarray | PIL.Image,
        Tensor | np.ndarray | PIL.Image | None
    ]:
        """
        It applies the transforms to the input and target.
        
        Args:
            input (Tensor | np.ndarray | PIL.Image): The input tensor to be
                transformed.
            target (Tensor | np.ndarray | PIL.Image | None): The target tensor
                to be transformed.
        
        Returns:
            The transformed input and target.
        """
        for t in self.transforms:
            input, target = t(input, target)
        return input, target
    
    def __repr__(self) -> str:
        """
        The function returns a string that contains the name of the class,
        and the string representation of each transform in the list of
        transforms.
        
        Returns:
            A string representation of the object.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor | np.ndarray | PIL.Image,
        target: Tensor | np.ndarray | PIL.Image | None = None,
        *args, **kwargs
    ) -> tuple[
        Tensor | np.ndarray | PIL.Image,
        Tensor | np.ndarray | PIL.Image | None
    ]:
        """
        It applies the transforms to the input and target.
        
        Args:
            input (Tensor | np.ndarray | PIL.Image): The input tensor to be
                transformed.
            target (Tensor | np.ndarray | PIL.Image | None): The target tensor
                to be transformed.
        
        Returns:
            The transformed input and target.
        """
        for t in self:
            input, target = t(input, target)
        return input, target

    
# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
