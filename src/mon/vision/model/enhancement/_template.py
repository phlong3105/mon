#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a template for implement enhancement models."""

from __future__ import annotations

__all__ = [

]

from mon.vision.typing import (
    ClassLabelsType, ConfigType, DictType, LossesType,
    MetricsType, ModelPhaseType, OptimizersType, PathType, WeightsType,
)
from torch import nn

from mon import core
from mon.vision import constant
from mon.vision.model.enhancement import base


# region Model

@constant.MODELS.register(name="template-model")
class TemplateModel(base.ImageEnhancementModel):
    """Template Model.
    
    See base class: :class:`base.ImageEnhancementModel`
    """
    
    cfgs = {
    
    }
    pretrained_weights = {
    
    }
    
    def __init__(
        self,
        cfg: ConfigType | None = "template.yaml",
        hparams: DictType | None = None,
        channels: int = 3,
        num_classes: int | None = None,
        classlabels: ClassLabelsType | None = None,
        weights: WeightsType = False,
        # For management
        name: str | None = "template",
        variant: str | None = None,
        fullname: str | None = "template",
        root: PathType = constant.RUN_DIR,
        project: str | None = None,
        # For training
        phase: ModelPhaseType = "training",
        loss: LossesType | None = None,
        metrics: MetricsType | None = None,
        optimizers: OptimizersType | None = None,
        debug: DictType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            cfg=cfg,
            hparams=hparams,
            channels=channels,
            num_classes=num_classes,
            classlabels=classlabels,
            weights=weights,
            name=name,
            variant=variant,
            fullname=fullname,
            root=root,
            project=project,
            phase=phase,
            loss=loss,
            metrics=metrics,
            optimizers=optimizers,
            debug=debug,
            verbose=verbose,
            *args, **kwargs
        )
    
    @property
    def cfg_dir(self) -> PathType:
        return core.Path(__file__).resolve().parent / "cfg"
    
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        pass
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        super().load_weights()

# endregion
