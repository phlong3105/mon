#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all annotations."""

from __future__ import annotations

__all__ = [
    "Annotation",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from mon import core

console = core.console


# region Base

class Annotation(ABC):
    """The base class for all annotation classes. An annotation instance
    represents a logical collection of data associated with a particular task.
    """
    
    @property
    @abstractmethod
    def data(self) -> list | None:
        """The annotation's data."""
        pass
    
    @property
    def nparray(self) -> np.ndarray | None:
        """The annotation's data as a :obj:`numpy.ndarray`."""
        data = self.data
        if isinstance(data, list):
            data = np.array([i for i in data if isinstance(i, int | float)])
        return data
    
    @property
    def tensor(self) -> torch.Tensor | None:
        """The annotation's data as a :obj:`torch.Tensor`."""
        data = self.data
        if isinstance(data, list):
            data = torch.Tensor([i for i in data if isinstance(i, int | float)])
        return data
    
    @staticmethod
    @abstractmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collate function used to fused input items together when using
		:obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A :obj:`list` of objects.
		"""
        pass
    
# endregion
