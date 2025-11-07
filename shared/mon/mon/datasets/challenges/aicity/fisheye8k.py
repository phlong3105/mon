#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the FishEye8K dataset."""

__all__ = [
    "FishEye8K",
]

from mon.core import rich
from ...core import *


@DATASETS.register(name="fisheye8k")
class FishEye8K(VisionDataset):
    """FishEye8K dataset."""
    
    root_name : str         = "fisheye8k"
    tasks     : list[Task]  = [Task.DETECT]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    classes   : Classes     = Classes([
        {"name": "bus",        "id": 0, "color": [140,  24, 143]},
        {"name": "bike",       "id": 1, "color": [122,  35,   2]},
        {"name": "car",        "id": 2, "color": [ 49,   3, 150]},
        {"name": "pedestrian", "id": 3, "color": [ 81, 120, 228]},
        {"name": "truck",      "id": 4, "color": [ 72, 153, 152]},
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
