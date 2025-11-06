#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DarkFace datasets."""

__all__ = [
    "DarkFace",
]

from mon.core import rich
from mon.datasets.core import *


@DATASETS.register(name="darkface")
class DarkFace(VisionDataset):
    """DarkFace dataset."""

    root_name : str         = "darkface"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = Classes([
        {"name": "face", "id": 0, "color": [ 81, 120, 228]},
    ])

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / f"{self.split_str}" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
