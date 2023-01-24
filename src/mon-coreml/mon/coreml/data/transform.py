#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base transformation class."""

from __future__ import annotations

__all__ = [
    "ComposeTransform", "Transform",
]

from abc import ABC, abstractmethod
from typing import Sequence, TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mon.coreml import constant

if TYPE_CHECKING:
    from mon.coreml.data.dataset import Dataset
    from mon.coreml.typing import TransformType


# region Transform

class Transform(nn.Module, ABC):
    """The base class for all transformations.
    
    Args:
        p: The probability determines whether the transformation is applied.
            Defaults to None mean process as normal.
    """
    
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if p is not None:
            assert 0.0 <= p <= 1.0
        self.p = p
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __call__(
        self,
        input  : torch.Tensor | np.ndarray,
        target : torch.Tensor | np.ndarray | None = None,
        dataset: Dataset                   | None = None,
    ) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None
    ]:
        """If the probability is greater than a random number, then applies
        transforms.
        
        Args:
            input: An input to be transformed.
            target: A target to be transformed. Defaults to None.
            dataset: A dataset containing the :param:`input` and :param:`target`
                which provides additional information for the transformation.
                Defaults to None.
        
        Returns:
            A tuple (input, target), where input and target have been
            transformed.
        """
        if self.p is None or torch.rand(1).item() <= self.p:
            return super().__call__(
                input   = input,
                target  = target,
                dataset = dataset,
            )
        return input, target
        
    @abstractmethod
    def forward(
        self,
        input  : torch.Tensor | np.ndarray,
        target : torch.Tensor | np.ndarray | None = None,
        dataset: Dataset                   | None = None,
    ) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None
    ]:
        """Transforms the :param:`input` and :param:`target`.
        
        Args:
            input: An input to be transformed.
            target: A target to be transformed. Defaults to None.
            dataset: A dataset containing the :param:`input` and :param:`target`
                which provides additional information for the transformation.
                Defaults to None.
        
        Returns:
            A tuple (input, target), where input and target have been
            transformed.
        """
        pass

# endregion


# region Composed Transforms

class ComposeTransform(nn.Sequential):
    """A sequence of :class:`Transformation` stacked together.

    Args:
        transforms: A list of :class:`Transformation` objects.

    Example:
        >>> transforms.Compose(transforms=[
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    Notes:
        In order to script the transformations, please use `torch.nn.Sequential`
        as below:
        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with
        `Tensor`, does not require `lambda` functions or `PIL.Image`.
    """
    
    def __init__(
        self,
        transforms: TransformType | Sequence[TransformType],
        *args, **kwargs
    ):
        if isinstance(transforms, dict):
            transforms = [v for k, v in transforms.items()]
        if not isinstance(transforms, list | tuple):
            transforms = [transforms]
        
        transforms = [
            constant.TRANSFORM.build(cfg=t)
            if isinstance(t, dict) else t for t in transforms
        ]
        if not all(isinstance(t, Transform) for t in transforms):
            raise TypeError(
                f"All items in `transforms` must be callable. "
                f"But got: {transforms}."
            )
        
        args = transforms + list(args)
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        input  : torch.Tensor | np.ndarray,
        target : torch.Tensor | np.ndarray | None = None,
        dataset: Dataset                   | None = None,
    ) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None
    ]:
        """Applies transforms to the :param:`input` and :param:`target`.
        
        Args:
            input: An input to be transformed.
            target: A target to be transformed. Defaults to None.
            dataset: A dataset containing the :param:`input` and :param:`target`
                which provides additional information for the transformation.
                Defaults to None.
        
        Returns:
            A tuple (input, target), where input and target have been
            transformed.
        """
        # for t in self.named_modules():
        #     input, target = t(input, target)
        return super().__call__(
            input   = input,
            target  = target,
            dataset = dataset,
        )
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
    def forward(
        self,
        input  : torch.Tensor | np.ndarray,
        target : torch.Tensor | np.ndarray | None = None,
        dataset: Dataset                   | None = None,
    ) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None
    ]:
        """Applies transforms to the :param:`input` and :param:`target`.
        
        Args:
            input: An input to be transformed.
            target: A target to be transformed. Defaults to None.
            dataset: A dataset containing the :param:`input` and :param:`target`
                which provides additional information for the transformation.
                Defaults to None.
        
        Returns:
            A tuple (input, target), where input and target have been
            transformed.
        """
        for t in self:
            input, target = t(
                input   = input,
                target  = target,
                dataset = dataset,
            )
        return input, target

# endregion
