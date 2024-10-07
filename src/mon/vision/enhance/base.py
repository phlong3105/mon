#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Image Enhancement Model.

This module implements the base class for enhancement models.
"""

from __future__ import annotations

__all__ = [
    "ImageEnhancementModel",
]

from abc import ABC

import cv2

from mon import core, nn
from mon.globals import Scheme
from mon.vision.model import VisionModel

console = core.console


# region Model

class ImageEnhancementModel(VisionModel, ABC):
    """The base class for all image enhancement models."""
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        if "image" not in datapoint:
            raise ValueError(f"The key ``'image'`` must be defined in the "
                             f"`datapoint`.")
        
        has_target = any(item in self.schemes for item in [Scheme.SUPERVISED]) and not self.predicting
        if has_target:
            if "ref_image" not in datapoint:
                raise ValueError(f"The key ``'ref_image'`` must be defined in "
                                 f"the `datapoint`.")
            
    def assert_outputs(self, outputs: dict) -> bool:
        if "enhanced" not in outputs:
            raise ValueError(f"The key ``'enhanced'`` must be defined in the "
                             f"`outputs`.")
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred   = outputs.get("enhanced")
        target = datapoint.get("ref_image")
        outputs["loss"] = self.loss(pred, target)
        # Return
        return outputs
    
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] = None
    ) -> dict:
        # Check
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Metrics
        pred    = outputs.get("enhanced")
        target  = datapoint.get("ref_image")
        results = {}
        if metrics is not None:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
        
    def log_images(
        self,
        epoch    : int,
        step     : int,
        data     : dict,
        extension: str = ".jpg"
    ):
        epoch    = int(epoch)
        step     = int(step)
        save_dir = self.debug_dir / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        image     =    data.get("image",    None)
        ref_image =    data.get("ref_image", None)
        outputs   =    data.get("outputs",  {})
        enhanced  = outputs.pop("enhanced", None)
        
        image        = list(core.to_image_nparray(image,     keepdim=False, denormalize=True))
        ref_image    = list(core.to_image_nparray(ref_image, keepdim=False, denormalize=True)) if ref_image is not None else None
        enhanced     = list(core.to_image_nparray(enhanced,  keepdim=False, denormalize=True))
        extra_images = {k: v for k, v in outputs.items() if core.is_image(v)}
        extra        = {
            k: list(core.to_image_nparray(v, keepdim=False, denormalize=True))
            for k, v in extra_images.items()
        } if extra_images else {}
        
        if len(image) != len(enhanced):
            raise ValueError(f"The number of `image` and `enhanced` must be "
                             f"the same, but got {len(image)} != {len(enhanced)}.")
        if ref_image is not None:
            if len(image) != len(ref_image):
                raise ValueError(f"The number of `image` and `ref_image` must "
                                 f"be the same, but got "
                                 f"{len(image)} != {len(ref_image)}.")
            
        for i in range(len(image)):
            if ref_image:
                combined = cv2.hconcat([image[i], enhanced[i], ref_image[i]])
            else:
                combined = cv2.hconcat([image[i], enhanced[i]])
            combined    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            output_path = save_dir / f"{i}{extension}"
            cv2.imwrite(str(output_path), combined)
            
            for k, v in extra.items():
                v_i = v[i]
                extra_path = save_dir / f"{i}_{k}{extension}"
                cv2.imwrite(str(extra_path), v_i)
        
# endregion
