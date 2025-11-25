#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GT-Snow dataset for image desnowing tasks."""

__all__ = [
    "GTSnow",
]

from ....core import *


@DATASETS.register(name="gtsnow")
class GTSnow(ImageDataset):
    """GTSnow dataset."""
    
    root_name : str         = "gtsnow"
    tasks     : list[Task]  = [Task.DESNOW]
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None
