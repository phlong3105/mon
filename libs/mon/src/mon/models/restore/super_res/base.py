#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for super-resolution models.
"""

from __future__ import annotations

__all__ = [
    "SuperResModel",
]

from abc import ABC, abstractmethod

from torch import nn, Tensor

from mon.core import IntOrTuple2, Size


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SuperResModel(nn.Module, ABC):
    """Base class for all super-resolution models."""

    # --- Callable & Context Manager ---
    @abstractmethod
    def forward(
        self,
        x_lr: Tensor,
        y_hr: Tensor | None = None,
        imgsz: Size | IntOrTuple2 | None = None,
        *args, **kwargs
    ) -> dict:
        """Forward the input through the model.

        Either ``imgsz`` or ``y_hr`` must be provided.

        Args:
            x_lr (Tensor): Low-resolution input image of shape (B, C, H0, W0)
                and values ranging from 0.0 to 1.0.
            y_hr (Tensor, optional): High-resolution guidance image of shape
                (B, C, H1, W1) and values ranging from 0.0 to 1.0.
                Defaults to None.
            imgsz (Size | IntOrTuple2, optional): Desired output image size.
                Can be a Size object or a tuple (height, width). Defaults to None.
        """
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
