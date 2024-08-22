#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for enhancement models."""

from __future__ import annotations

__all__ = [
    "ImageEnhancementModel",
]

from abc import ABC

import cv2

from mon import core, nn
from mon.globals import Scheme, ZOO_DIR
from mon.vision.model import VisionModel

console = core.console


# region Model

class ImageEnhancementModel(VisionModel, ABC):
    """The base class for all image enhancement models."""
    
    zoo_dir: core.Path = ZOO_DIR / "vision" / "enhance"
    
    # region Forward Pass
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        assert "image" in datapoint, "The key ``'image'`` must be defined in the `datapoint`."
        
        has_target = any(item in self.schemes for item in [Scheme.SUPERVISED])
        if has_target:
            assert "hq_image" in datapoint, "The key ``'hq_image'`` must be defined in the `datapoint`."
            
    def assert_outputs(self, outputs: dict) -> bool:
        assert "enhanced" in outputs, "The key ``'enhanced'`` must be defined in the `outputs`."
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred    = outputs.get("enhanced")
        target  = datapoint.get("hq_image")
        outputs["loss"] = self.loss(pred, target) if self.loss else None
        # Return
        return outputs
    
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] | None = None
    ) -> dict:
        # Check
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Metrics
        pred    = outputs.get("enhanced")
        target  = datapoint.get("hq_image")
        results = {}
        if metrics is not None:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
    
    # endregion
    
    # region Logging
    
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
        
        image    =    data.get("image",    None)
        hq_image =    data.get("hq_image", None)
        outputs  =    data.get("outputs",  {})
        enhanced = outputs.pop("enhanced", None)
        
        image        = list(core.to_image_nparray(image,    keepdim=False, denormalize=True))
        hq_image     = list(core.to_image_nparray(hq_image, keepdim=False, denormalize=True)) if hq_image is not None else None
        enhanced     = list(core.to_image_nparray(enhanced, keepdim=False, denormalize=True))
        extra_images = {k: v for k, v in outputs.items() if core.is_image(v)}
        extra        = {
            k: list(core.to_image_nparray(v, keepdim=False, denormalize=True))
            for k, v in extra_images.items()
        } if extra_images else {}
        
        assert len(image) == len(enhanced)
        if hq_image:
            assert len(image) == len(hq_image)
        
        for i in range(len(image)):
            if hq_image:
                combined = cv2.hconcat([image[i], enhanced[i], hq_image[i]])
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
    
# endregion
