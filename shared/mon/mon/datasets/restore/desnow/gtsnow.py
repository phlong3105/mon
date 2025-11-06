#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GT-Snow datasets."""

__all__ = [
    "GTSnow",
]

import os

from mon.core import pathlib, rich
from mon.datasets.core import *


@DATASETS.register(name="gtsnow")
class GTSnow(VisionDataset):
    """GTSnow dataset."""
    
    root_name : str         = "gtsnow"
    tasks     : list[Task]  = [Task.DESNOW]
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image and ref annotations."""
        patterns = [self.root / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
