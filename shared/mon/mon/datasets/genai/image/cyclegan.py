#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the CycleGAN dataset for image-to-image translation ;
tasks.
"""

__all__ = [
    "CycleGANDataset",
]

from mon.core import Path, rich
from mon.training import albumentations as A
from ...core import *


@DATASETS.register(name="cyclegan_dataset")
class CycleGANDataset(VisionDualDomainDataset):
    """Cycle-GAN dataset."""
    
    root_name   : str         = "cyclegan"
    subsets     : list[str]   = [
        "ae_photos", "apple2orange", "cezanne2photo", "facades", "grumpifycat",
        "horse2zebra", "iphone2dslr_flower", "maps", "mini", "monet2photo",
        "summer2winter_yosemite", "ukiyoe2photo", "vangogh2photo"
    ]
    tasks       : list[Task]  = [Task.IMG2IMG]
    splits      : list[Split] = [Split.TEST]
    modalities_A: Modalities  = {
        "image_A": Modality(name="image_A", type="image", module=Image, train=True, test=True, primary=True),
    }
    modalities_B: Modalities  = {
        "image_B": Modality(name="image_B", type="image", module=Image, train=True, test=True, primary=True),
    }
    classes     : Classes     = None
    
    def __init__(
        self,
        root     : Path,
        subset   : str       = "monet2photo",
        split    : Split     = Split.TRAIN,
        transform: A.Compose = None,
        serial   : bool      = False,
        verbose  : bool      = True,
        *args, **kwargs
    ):
        if subset not in self.subsets:
            raise ValueError(f"``subset`` must be one of: {self.subsets}, but got {subset}.")
        
        self.subset = subset
        self.serial = serial
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            verbose   = verbose,
            *args, **kwargs
        )
    
    def list_primary_data_A(self) -> list:
        patterns = [self.root / self.subset / self.split_str / "image_A"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
    
    def list_primary_data_B(self) -> list:
        patterns = [self.root / self.subset / self.split_str / "image_B"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
