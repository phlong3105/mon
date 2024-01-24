#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base classes for all labels."""

from __future__ import annotations

__all__ = [
    "Label",
]

from abc import ABC, abstractmethod

import numpy as np
import torch

from mon import core

console = core.console


# region Base

class Label(ABC):
    """The base class for all label classes. A label instance represents a
    logical collection of data associated with a particular task.
    """
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    """
    
    @property
    @abstractmethod
    def data(self) -> list | None:
        """The label's data."""
        pass
    
    @property
    def nparray(self) -> np.ndarray | None:
        """The label's data as a :class:`numpy.ndarray`."""
        data = self.data
        if isinstance(data, list):
            data = np.array([i for i in data if isinstance(i, int | float)])
        return data
    
    @property
    def tensor(self) -> torch.Tensor | None:
        """The label's data as a :class:`torch.Tensor`."""
        data = self.data
        if isinstance(data, list):
            data = torch.Tensor([i for i in data if isinstance(i, int | float)])
        return data

# endregion
