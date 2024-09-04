#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Classification Model.

This module implements the base model class for classification models.
"""

from __future__ import annotations

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import Scheme, Task
from mon.vision.model import VisionModel

console = core.console


# region Model

class ImageClassificationModel(VisionModel, ABC):
    """The base class for all image classification models."""
    
    tasks: list[Task] = [Task.CLASSIFY]
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        if "image" not in datapoint:
            raise ValueError(f"The key ``'image'`` must be defined in the "
                             f"`datapoint`.")
        
        has_target = any(item in self.schemes for item in [Scheme.SUPERVISED])
        if has_target:
            if "class_id" not in datapoint:
                raise ValueError(f"The key ``'class_id'`` must be defined in "
                                 f"the `datapoint`.")
            
    def assert_outputs(self, outputs: dict) -> bool:
        if "logits" not in outputs:
            raise ValueError(f"The key ``'logits'`` must be defined in the "
                             f"`outputs`.")
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred    = outputs.get("logits")
        target  = datapoint.get("class_id")
        outputs["loss"] = self.loss(pred, target) if self.loss else None
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
        pred    = outputs.get("logits")
        target  = datapoint.get("class_id")
        results = {}
        if metrics is not None:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
    
# endregion
