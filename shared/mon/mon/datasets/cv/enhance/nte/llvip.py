#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the LLVIP dataset for nighttime object detection tasks.

References:
    - Paper: "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision," ICCV 2021.
    - Data: https://github.com/bupt-ai-cz/LLVIP
"""

__all__ = [
    "LLVIP",
]

from ....core import *


@DATASETS.register(name="llvip")
class LLVIP(ImageDataset):
    """LLVIP dataset."""
    
    root_name : str         = "llvip"
    tasks     : list[Task]  = [Task.NTE, Task.DETECT]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image"   : Modality(name="image",      type="image", module=Image,              train=True, test=True, primary=True),
        "depth"   : Modality(name=DepthName,    type="image", module=DefaultDepthMap,    train=True, test=True),
        "infrared": Modality(name=InfraredName, type="mask",  module=DefaultInfraredMap, train=True, test=True),
    }
    classes   : Classes     = None
