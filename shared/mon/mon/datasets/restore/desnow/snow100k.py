#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Snow100K datasets."""

__all__ = [
    "Snow100K",
]

from mon.core import rich
from mon.datasets.core import *


@DATASETS.register(name="snow100k")
class Snow100K(VisionDataset):
    """Snow100K dataset."""

    root_name : str         = "snow100k"
    tasks     : list[Task]  = [Task.DESNOW]
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None
    
    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "lq"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
        
        return images
