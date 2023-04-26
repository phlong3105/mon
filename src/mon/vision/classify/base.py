#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base model class for classification models."""

from __future__ import annotations

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

import torch

from mon.coreml import model
from mon.foundation import pathlib
from mon.vision import visualize


# region Model

class ImageClassificationModel(model.Model, ABC):
    """The base class for all image classification models.
    
    See Also: :class:`mon.coreml.model.Model`.
    """
    
    @property
    def config_dir(self) -> pathlib.Path:
        current_file = pathlib.Path(__file__).absolute()
        config_dir   = current_file.parent / "config"
        return config_dir
    
    def init_weights(self, m: torch.nn.Module):
        """Initialize model's weights."""
        pass
    
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
        filepath     : pathlib.Path | None = None,
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
        save_config = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        } if save else None
        visualize.imshow_classification(
            winname     = self.fullname, # self.phase.value,
            image       = input,
            pred        = pred,
            target      = target,
            scale       = 2,
            save_config = save_config,
            max_n       = max_n,
            nrow        = nrow,
            wait_time   = wait_time,
        )

# endregion
