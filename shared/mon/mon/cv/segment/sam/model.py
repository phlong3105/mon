#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Ultralytics SAM model for segmentation.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

__all__ = [
    "SAM",
    "SAM2",
    "SAM2_B",
    "SAM2_L",
    "SAM2_S",
    "SAM2_T",
    "SAM_B",
    "SAM_L",
]

from typing import Any

import box
from ultralytics import SAM as SAM_

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- SAM -----
class SAM(SAM_, nn.ModelMixin):
    """Ultralytics SAM model for segmentation.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "sam"
    name     : str          = "sam"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "sa1b", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)


@MODELS.register(name="sam_b", arch="sam")
class SAM_B(SAM):
    
    arch: str  = "sam"
    name: str  = "sam_b"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/sam/sam_b/sa1b/sam_b_sa1b.pt",
            "num_classes": None,
        },
    })


@MODELS.register(name="sam_l", arch="sam")
class SAM_L(SAM):
    
    arch: str  = "sam"
    name: str  = "sam_l"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/sam/sam_l/sa1b/sam_l_sa1b.pt",
            "num_classes": None,
        },
    })


# ----- SAM2 -----
class SAM2(SAM_, nn.ModelMixin):
    
    arch: str  = "sam2"
    name: str  = "sam2"
    zoo : dict = box.Box()
    
    def __init__(self, weights: Any = "sav", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)


@MODELS.register(name="sam2_t", arch="sam2")
class SAM2_T(SAM):
    
    arch: str  = "sam2"
    name: str  = "sam2_t"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/sam2/sam2.1_t/sav/sam2.1_t_sav.pt",
            "num_classes": None,
        },
    })
    

@MODELS.register(name="sam2_s", arch="sam2")
class SAM2_S(SAM):
    
    arch: str  = "sam2"
    name: str  = "sam2_s"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/sam2/sam2.1_s/sav/sam2.1_s_sav.pt",
            "num_classes": None,
        },
    })
    
    
@MODELS.register(name="sam2_b", arch="sam2")
class SAM2_B(SAM):
    
    arch: str  = "sam2"
    name: str  = "sam2_b"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/sam2/sam2.1_b/sav/sam2.1_b_sav.pt",
            "num_classes": None,
        },
    })


@MODELS.register(name="sam2_l", arch="sam2")
class SAM2_L(SAM):
    
    arch: str  = "sam2"
    name: str  = "sam2_l"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/sam2/sam2.1_l/sav/sam2.1_l_sav.pt",
            "num_classes": None,
        },
    })
