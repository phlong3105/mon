#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ours custom TSD (Traffic Surveillance Dataset) datasets."""

__all__ = [
    "TSDME",
]

from mon.core import rich
from ...core import *


@DATASETS.register(name="tsdme")
class TSDME(ImageDataset):
    """TSD-ME dataset."""
    
    _subset    : str         = "tsd"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : ClassList   = None
    
    def _load_primary_data(self) -> list:
        if self.split == Split.TRAIN:
            patterns = [
                self.root / "me" / self.split_str / "image",
                self.root / "me" / "extra"        / "image",
            ]
        else:
            patterns = [
                self.root / "me" / self.split_str / "image",
            ]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
    
        return images
