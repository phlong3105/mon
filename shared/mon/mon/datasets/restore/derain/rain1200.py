#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Rain1200 datasets."""

__all__ = [
    "Rain1200",
]

from mon.core import rich
from mon.datasets.core import *


@DATASETS.register(name="rain1200")
class Rain1200(VisionDataset):
    """Rain1200 dataset."""

    root_name : str         = "rain1200"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None
    
    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / self.split_str / "light"  / "image",
                self.root / self.split_str / "medium" / "image",
                self.root / self.split_str / "heavy"  / "image",
            ]
        else:
            patterns = [
                self.root / self.split_str / "image",
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
