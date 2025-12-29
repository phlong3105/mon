#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MSEC datasets.

This module implements the Multi-Scale Exposure Correction (MSEC) dataset for
exposure correction.

References:
    - Paper: "Learning Multi-Scale Photo Exposure Correction," CVPR 2021.
    - Code: https://github.com/mahmoudnafifi/Exposure_Correction
"""

__all__ = [
    "MSEC",
]

from mon.core import rich
from ....meta import *


@DATASETS.register(name="msec")
class MSEC(ImageDataset):
    """MSEC dataset."""
    
    _subset    : str         = "msec"
    _tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image"        : Modality(name="image_ev_0",    type="image", module=Image, train=True, test=True, primary=True),
        "image_ev_n1.5": Modality(name="image_ev_n1.5", type="image", module=Image, train=True, test=True),
        "image_ev_n1"  : Modality(name="image_ev_n1",   type="image", module=Image, train=True, test=True),
        "image_ev_0"   : Modality(name="image_ev_0",    type="image", module=Image, train=True, test=True),
        "image_ev_p1"  : Modality(name="image_ev_p1",   type="image", module=Image, train=True, test=True),
        "image_ev_p1.5": Modality(name="image_ev_p1.5", type="image", module=Image, train=True, test=True),
        "ref"          : Modality(name="ref_c",         type="image", module=Image, train=True, test=True),
    }
    _classlist : ClassList   = None
    
    def __init__(self, lr: bool = True, *args, **kwargs):
        """Initialize a new instance.
        
        Args:
            lr (bool): If True, use low-resolution versions of the images.
                Default is True.
        """
        self.lr = lr
        super().__init__(*args, **kwargs)
    
    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
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
                        images.append(Image(data=path, root=pattern))
        
        return images
