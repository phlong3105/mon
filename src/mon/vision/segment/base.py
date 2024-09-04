#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Segmentation Model.

This module implements the base class for segmentation models.
"""

from __future__ import annotations

__all__ = [
    "SegmentationModel",
]

from abc import ABC

import cv2

from mon import core, nn
from mon.globals import Scheme, Task
from mon.vision.model import VisionModel

console = core.console


# region Model

class SegmentationModel(VisionModel, ABC):
    """The base class for all segmentation models."""
    
    tasks: list[Task] = [Task.SEGMENT]
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        if "image" not in datapoint:
            raise ValueError(f"The key ``'image'`` must be defined in the "
                             f"`datapoint`.")
        
        has_target = any(item in self.schemes for item in [Scheme.SUPERVISED])
        if has_target:
            if "semantic" not in datapoint:
                raise ValueError(f"The key ``'semantic'`` must be defined in "
                                 f"the `datapoint`.")
            
    def assert_outputs(self, outputs: dict) -> bool:
        if "semantic" not in outputs:
            raise ValueError(f"The key ``'semantic'`` must be defined in the "
                             f"`outputs`.")
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred   = outputs.get("semantic")
        target = datapoint.get("semantic")
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
        pred    = outputs.get("semantic")
        target  = datapoint.get("semantic")
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
        
        image         =    data.get("image",    None)
        tar_semantic  =    data.get("semantic", None)
        outputs       =    data.get("outputs",  {})
        pred_semantic = outputs.pop("semantic", None)
        
        image         = list(core.to_image_nparray(image,         keepdim=False, denormalize=True))
        tar_semantic  = list(core.to_image_nparray(tar_semantic,  keepdim=False, denormalize=True)) if tar_semantic is not None else None
        pred_semantic = list(core.to_image_nparray(pred_semantic, keepdim=False, denormalize=True))
        extra_images  = {k: v for k, v in outputs.items() if core.is_image(v)}
        extra         = {
            k: list(core.to_image_nparray(v, keepdim=False, denormalize=True))
            for k, v in extra_images.items()
        } if extra_images else {}
        
        if len(image) != len(pred_semantic):
            raise ValueError(f"The number of `images` and `pred_semantic` must "
                             f"be the same, but got "
                             f"{len(image)} != {len(pred_semantic)}.")
        if tar_semantic is not None:
            if len(image) != len(tar_semantic):
                raise ValueError(f"The number of `images` and `tar_semantic` "
                                 f"must be the same, but got "
                                 f"{len(image)} != {len(tar_semantic)}.")
        
        for i in range(len(image)):
            if tar_semantic:
                combined = cv2.hconcat([image[i], pred_semantic[i], tar_semantic[i]])
            else:
                combined = cv2.hconcat([image[i], pred_semantic[i]])
            combined    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            output_path = save_dir / f"{i}{extension}"
            cv2.imwrite(str(output_path), combined)
            
            for k, v in extra.items():
                v_i = v[i]
                extra_path = save_dir / f"{i}_{k}{extension}"
                cv2.imwrite(str(extra_path), v_i)
                
# endregion
