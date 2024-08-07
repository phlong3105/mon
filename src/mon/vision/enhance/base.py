#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for enhancement models."""

from __future__ import annotations

__all__ = [
    "ImageEnhancementModel",
]

from abc import ABC

import cv2
import torch

from mon import core, nn
from mon.globals import ZOO_DIR

console = core.console


# region Model

class ImageEnhancementModel(nn.Model, ABC):
    """The base class for all image enhancement models.
    
    See Also: :class:`nn.Model`.
    """
    
    zoo_dir: core.Path = ZOO_DIR / "vision" / "enhance"
    
    def log_image(
        self,
        epoch    : int,
        step     : int,
        input    : torch.Tensor,
        pred     : torch.Tensor,
        target   : torch.Tensor | None = None,
        extra    : dict         | None = None,
        extension: str = ".jpg"
    ):
        epoch    = int(epoch)
        step     = int(step)
        save_dir = self.debug_dir / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        input    = list(core.to_image_nparray(input,  keepdim=False, denormalize=True))
        pred     = list(core.to_image_nparray(pred,   keepdim=False, denormalize=True))
        target   = list(core.to_image_nparray(target, keepdim=False, denormalize=True)) if target is not None else None
        extra    = {
            k: list(core.to_image_nparray(v, keepdim=False, denormalize=True))
            for k, v in extra.items()
        } if extra is not None else {}
        
        assert len(input) == len(pred)
        if target is not None:
            assert len(input) == len(target)
        
        for i in range(len(input)):
            if target is not None:
                combined = cv2.hconcat([input[i], pred[i], target[i]])
            else:
                combined = cv2.hconcat([input[i], pred[i]])
            combined    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            output_path = save_dir / f"{i}{extension}"
            cv2.imwrite(str(output_path), combined)
            
            for k, v in extra.items():
                extra_path = save_dir / f"{i}_{k}{extension}"
                cv2.imwrite(str(extra_path), cv2.cvtColor(v[i], cv2.COLOR_RGB2BGR))
            
# endregion
