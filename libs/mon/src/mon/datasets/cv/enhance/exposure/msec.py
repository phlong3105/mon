#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MSEC datasets.

This module provides the Multi-Scale Exposure Correction (MSEC) dataset for
exposure correction.

References:
    - Paper: "Learning Multi-Scale Photo Exposure Correction," CVPR 2021.
    - Code: https://github.com/mahmoudnafifi/Exposure_Correction
"""

from __future__ import annotations

__all__ = [
    "MSEC",
]


from ....api import *


@DATASETS.register()
class MSEC(ImageDataset, RegistrableMixin):
    """MSEC dataset."""
    
    _name      : str         = "msec"
    _tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image"        : Modality(
            name    = "image_ev_0",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "image_ev_n1.5": Modality(
            name    = "image_ev_n1.5",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_n1"  : Modality(
            name    = "image_ev_n1",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_0"   : Modality(
            name    = "image_ev_0",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_p1"  : Modality(
            name    = "image_ev_p1",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_p1.5": Modality(
            name    = "image_ev_p1.5",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "ref"          : Modality(
            name    = "ref_c",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None
    
    # --- Lifecycle & Initialization ---
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
        # Determine root directory based on resolution preference
        base_dir = "msec_lr" if self.lr else "msec"
        pattern  = self.root / base_dir / self.split_str / "image_ev_0"
        
        images   = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))
        
        return images
