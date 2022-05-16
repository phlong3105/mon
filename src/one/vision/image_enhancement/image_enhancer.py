#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Enhancer.
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

import torch.nn as nn
from torch import Tensor

from one.core import Indexes
from one.core import Tensors
from one.imgproc import imshow_plt
from one.nn import BaseModel

__all__ = [
    "ImageEnhancer"
]


# MARK: - ImageEnhancer

class ImageEnhancer(BaseModel, metaclass=ABCMeta):
    """Image Enhancer is a base class for all end-to-end trainable image
    enhancement models. End-to-End means that the whole network can be trained
    end-to-end, without having to freeze some parts of the network during
    training.
    
    Usually, we add functions such as:
    - Define main components.
    - Forward pass with custom loss and metrics.
    - Result visualization.
    
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
        self.head	 : Optional[nn.Module] = None
        
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
    
    def forward(
        self, x: Tensor, augment: bool = False, *args, **kwargs
    ) -> Tensor:
        """Forward pass. This is the primary `forward` function of the model.
        It supports augmented inference.
        
        In this function, we perform test-time augmentation and pass the
        transformed input to `forward_once()`.

        Args:
            x (Tensor[B, C, H, W]):
                Input.
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
			x (Tensor[B, C, H, W]):
				Input.

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
            x (Tensor[B, C, H, W]):
                Input.
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
            elif out_indexes is None or out_indexes == -1:
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
        """Draw `result` over input image.

        Args:
            x (Tensor, optional):
                Low quality images.
            y (Tensor, optional):
                High quality images.
            yhat (Tensor, optional):
                Enhanced images.
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
        # NOTE: Prepare images
        results = {}
        if x is not None:
            results["low"] = x
        if y is not None:
            results["high"] = y
        if yhat is not None:
            results["enhance"] = yhat
            
        filepath = self.debug_image_filepath if filepath is None else filepath
        save_cfg = {
            "filepath"  : filepath,
            "pil_kwargs": dict(quality=image_quality)
        }
        imshow_plt(
            images     = results,
            scale      = 2,
            save_cfg   = save_cfg,
            verbose    = verbose,
            show_max_n = show_max_n,
            wait_time  = wait_time
        )
