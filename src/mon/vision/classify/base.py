#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base model class for classification models."""

from __future__ import annotations

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import Task, ZOO_DIR, Scheme
from mon.nn import metric as M

console = core.console


# region Model

class ImageClassificationModel(nn.Model, ABC):
    """The base class for all image classification models.
    
    See Also: :class:`mon.nn.Model`.
    """
    
    tasks  : list[Task] = [Task.CLASSIFY]
    zoo_dir: core.Path  = ZOO_DIR / "vision" / "classify"
    
    @classmethod
    def assert_datapoint(cls, datapoint: dict) -> bool:
        assert datapoint.get("image", None), \
            "The key ``'image'`` must be defined in the :param:`datapoint`."
        
        has_target = any(item in cls.schemes for item in [Scheme.SUPERVISED])
        if has_target:
            assert datapoint.get("class_id", None), \
                "The key ``'class_id'`` must be defined in the :param:`datapoint`."
    
    @classmethod
    def assert_outputs(cls, outputs: dict) -> bool:
        assert outputs.get("logits", None), \
            "The key ``'logits'`` must be defined in the :param:`outputs`."
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Loss
        pred   = outputs.get("logits")
        target = datapoint.get("class_id")
        outputs["loss"] = self.loss(pred, target) if self.loss else None
        # Return
        return outputs
    
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[M.Metric] | None = None
    ) -> dict:
        # Check
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Metrics
        pred    = outputs.get("logits")
        target  = datapoint.get("class_id")
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
    
# endregion
