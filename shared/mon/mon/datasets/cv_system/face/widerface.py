#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the WiderFace dataset for face detection."""

__all__ = [
    "WiderFace",
    "WiderFaceVal",
]

from mon.core import rich
from ...core import *


@DATASETS.register(name="widerface")
class WiderFace(VisionDataset):
    """WiderFace dataset."""
    
    root_name : str         = "widerface"
    tasks     : list[Task]  = [Task.DETECT]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    classes   : Classes     = Classes([
        {"name": "face", "id": 0, "color": [ 81, 120, 228]},
    ])

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
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


@DATASETS.register(name="widerfaceval")
class WiderFaceVal(WiderFace):
    """WiderFace-Val subset."""

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "val" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
