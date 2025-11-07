#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the LightEffect dataset for deflaring tasks."""

__all__ = [
    "LightEffect",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="lighteffect")
class LightEffect(VisionDataset):
    """LightEffect dataset."""
    
    root_name : str         = "lighteffect"
    tasks     : list[Task]  = [Task.DEFLARE]
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    classes   : Classes     = None

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
