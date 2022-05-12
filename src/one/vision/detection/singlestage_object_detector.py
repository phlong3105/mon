#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Single Stage Object Detector.
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

from torch import nn
from torch import Tensor

from one.core import Indexes
from one.core import Tensors
from one.nn import BaseModel

__all__ = [
    "SingleStageObjectDetector",
]


# MARK: - SingleStageObjectDetector

class SingleStageObjectDetector(BaseModel, metaclass=ABCMeta):
    """Single Stage Object Detector.

    Attributes:
        backbone (nn.Module):
            Features extraction module.
        neck (nn.Module, optional):
            Neck module. Default: `None`.
        head (nn.Module):
            Head module.
    """

    # MARK: Magic Functions
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone: Optional[nn.Module] = None
        self.neck	 : Optional[nn.Module] = None
        self.head    : Optional[nn.Module] = None

    # MARK: Properties

    @property
    def with_backbone(self) -> bool:
        """Return whether if the `backbone` has been defined."""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_neck(self) -> bool:
        """Return whether if the `neck` has been defined."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Return whether if the `head` has been defined."""
        return hasattr(self, "head") and self.head is not None
    
    # MARK: Forward Pass

    def forward(self, x: Tensor, augment: bool = False, *args, **kwargs) -> Tensor:
        """Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].
            augment (bool):
                Augmented inference. Default: `False`.
                
        Returns:
            yhat (Tensor):
                Predictions.
        """
        if augment:
            # NOTE: For now just forward the input. Later, we will implement
            # the test-time augmentation for image classification
            return self.forward_once(x=x, *args, **kwargs)
        else:
            return self.forward_once(x=x, *args, **kwargs)
    
    @abstractmethod
    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass once. Implement the logic for a single forward pass.

		Args:
			x (Tensor):
				Input of shape [B, C, H, W].

		Returns:
			yhat (Tensor):
				Predictions.
		"""
        pass

    def forward_features(
        self, x: Tensor, out_indexes: Optional[Indexes] = None
    ) -> Tensors:
        """Forward pass for features extraction.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].
            out_indexes (Indexes, optional):
                List of layers' indexes to extract features. This is called
                in `forward_features()` and is useful when the model
                is used as a component in another model.
                - If is a `tuple` or `list`, return an array of features.
                - If is a `int`, return only the feature from that layer's
                  index.
                - If is `-1`, return the last layer's output.
                Default: `None`.
        """
        out_indexes = self.out_indexes if out_indexes is None else out_indexes
        if not self.with_backbone:
            raise ValueError(f"`backbone` has not been defined.")
    
        yhat = []
        for idx, m in enumerate(self.backbone.children()):
            x = m(x)
            if isinstance(out_indexes, (tuple, list)) and (idx in out_indexes):
                yhat.append(x)
            elif isinstance(out_indexes, int) and (idx == out_indexes):
                return x
            else:
                yhat = x
        return yhat

    # MARK: Visualization

    def show_results(
        self,
        x            : Optional[Tensor] = None,
        y            : Optional[Tensor] = None,
        yhat         : Optional[Tensor] = None,
        filepath     : Optional[str]    = None,
        image_quality: int              = 95,
        verbose      : bool             = False,
        show_max_n   : int              = 8,
        wait_time    : float            = 0.01,
        *args, **kwargs
    ):
        """Draw results.

        Args:
            x (Tensor, optional):
                Input.
            y (Tensor, optional):
                Ground-truth.
            yhat (Tensor, optional):
                Predictions.
            filepath (str, optional):
                File path to save the debug result.
            image_quality (int):
                Image quality to be saved. Default: `95`.
            verbose (bool):
                If `True` shows the results on the screen. Default: `False`.
            show_max_n (int):
                Maximum debugging items to be shown. Default: `8`.
            wait_time (float):
                Pause some times before showing the next image.
        """
        pass
