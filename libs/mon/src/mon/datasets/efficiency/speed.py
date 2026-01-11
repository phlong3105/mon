#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Speed benchmarking datasets.

This module implements datasets for efficiency benchmarking.
"""

__all__ = [
    "Speed1K",
]

from ..api import *


@DATASETS.register()
class Speed10(ImageDataset, RegistrableMixin):
    """Speed10 dataset."""
    
    _name      : str         = "speed10"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "speed10"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = None
    

@DATASETS.register()
class Speed1K(ImageDataset, RegistrableMixin):
    """Speed1K dataset."""

    _name      : str         = "speed1k"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "speed1k"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = None
