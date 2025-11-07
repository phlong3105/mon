#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Multi-Scale Exposure Correction (MSEC) dataset for
exposure correction tasks.

References:
    - Paper: "Learning Multi-Scale Photo Exposure Correction," CVPR 2021.
    - Code: https://github.com/mahmoudnafifi/Exposure_Correction
"""

__all__ = [
    "MSEC",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="msec")
class MSEC(VisionDataset):
    """MSEC dataset."""
    
    root_name : str         = "msec"
    tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image"        : Modality(name="image_ev_0",    type="image", module=Image, train=True, test=True, primary=True),
        "image_ev_n1.5": Modality(name="image_ev_n1.5", type="image", module=Image, train=True, test=True),
        "image_ev_n1"  : Modality(name="image_ev_n1",   type="image", module=Image, train=True, test=True),
        "image_ev_0"   : Modality(name="image_ev_0",    type="image", module=Image, train=True, test=True),
        "image_ev_p1"  : Modality(name="image_ev_p1",   type="image", module=Image, train=True, test=True),
        "image_ev_p1.5": Modality(name="image_ev_p1.5", type="image", module=Image, train=True, test=True),
        "ref"          : Modality(name="ref_c",         type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None
    
    def __init__(self, lr: bool = True, *args, **kwargs):
        self.lr = lr
        super().__init__(*args, **kwargs)
    
    def list_primary_data(self) -> list:
        if self.lr:
            patterns = [self.root / "msec_lr" / self.split_str / "image_ev_0"]
        else:
            patterns = [self.root / "msec"    / self.split_str / "image_ev_0"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
        
        return images
