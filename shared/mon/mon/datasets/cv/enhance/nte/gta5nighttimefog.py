#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the GTA5NighttimeFog dataset for nighttime image dehazing.

References:
    - Data: https://github.com/jinyeying/nighttime_dehaze
"""

__all__ = [
    "GTA5NighttimeFog",
]

from ....core import *


# ----- Dataset -----
@DATASETS.register(name="gta5nighttimefog")
class GTA5NighttimeFog(ImageDataset):
    """GTA5NighttimeFog dataset."""
    
    name      : str         = "gta5nighttimefog"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(DepthName,    type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",   type="image", module=Image,           train=True, test=True),
    }
    classes   : Classes     = None
