#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base model class for classification models."""

from __future__ import annotations

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

import torch
from torch import nn

from mon import coreml
from mon.vision import visualize
from mon.vision.typing import DictType, PathType


class ImageClassificationModel(coreml.Model, ABC):
    """The base class for all image classification models.
    
    See details at: :class:`coreml.Model`
    """
    
    def parse_model(
        self,
        d      : dict      | None = None,
        ch     : list[int] | None = None,
        hparams: DictType  | None = None,
    ) -> tuple[nn.Sequential, list[int], list[dict]]:
        """Build the model. You have 2 options for building a model: (1) define
        each layer manually, or (2) build model automatically from a config
        dictionary.
        
        Either way, each layer should have the following attributes:
            - i: index of the layer.
            - f: from, i.e., the current layer receives output from the f-th
                 layer. For example: -1 means from the previous layer; -2 means
                 from 2 previous layers; [99, 101] means from the 99th and 101st
                 layers. This attribute is used in forward pass.
            - t: type of the layer using this script:
                 t = str(m)[8:-2].replace("__main__.", "")
            - np: number of parameters using the following script:
              np = sum([x.numel() for x in m.parameters()])
        
        Args:
            d: Model definition dictionary. Default to None means building the
                model manually.
            ch: The first layer's input channels. If given, it will be used to
                further calculate the next layer's input channels. Defaults to
                None means defines each layer in_ and out_channels manually.
            hparams: Layer's hyperparameters. They are used to change the values
                of :param:`args`. Usually used in grid search or random search
                during training. Defaults to None.
            
        Return:
            A Sequential model.
            A list of layer index to save the features during forward pass.
            A list of layer's info for debugging.
        """
        return coreml.parse_model(d=d, ch=ch, hparams=hparams)
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : bool = False,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape [B, C, H, W].
            augment: If True, perform test-time augmentation. Defaults to False.
            profile: If True, Measure processing time. Defaults to False.
            out_index: Return specific layer's output from :param:`out_index`.
                Defaults to -1 means the last layer.
            
        Return:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            return self.forward_once(
                input   = input,
                profile = profile,
                *args, **kwargs
            )
        else:
            return self.forward_once(
                input   = input,
                profile = profile,
                *args, **kwargs
            )
    
    def show_results(
        self,
        input        : torch.Tensor | None = None,
        target	     : torch.Tensor | None = None,
        pred		 : torch.Tensor | None = None,
        filepath     : PathType     | None = None,
        image_quality: int                 = 95,
        max_n        : int          | None = 8,
        nrow         : int          | None = 8,
        wait_time    : float               = 0.01,
        save         : bool                = False,
        verbose      : bool                = False,
        *args, **kwargs
    ):
        """Show results.

        Args:
            input: An input.
            target: A ground-truth.
            pred: A prediction.
            filepath: A path to save the debug result.
            image_quality: The image quality to be saved. Defaults to 95.
            max_n: Show max n items if :param:`input` has a batch size of more
                than :param:`max_n` items. Defaults to None means show all.
            nrow: The maximum number of items to display in a row. The final
                grid size is (n / nrow, nrow). If None, then the number of items
                in a row will be the same as the number of items in the list.
                Defaults to 8.
            wait_time: Wait for some time (in seconds) to display the figure
                then reset. Defaults to 0.01.
            save: Save debug image. Defaults to False.
            verbose: If True shows the results on the screen. Defaults to False.
        """
        save_cfg = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        } if save else None
        visualize.imshow_classification(
            winname   = self.fullname,  # self.phase.value,
            image     = input,
            pred      = pred,
            target    = target,
            scale     = 2,
            save_cfg  = save_cfg,
            max_n     = max_n,
            nrow      = nrow,
            wait_time = wait_time,
        )
