#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LEDLight dataset for deflaring tasks."""

__all__ = [
    "LEDLight",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="ledlight")
class LEDLight(VisionDataset):
    """LEDLight dataset."""
    
    root_name : str         = "ledlight"
    tasks     : list[Task]  = [Task.DEFLARE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
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
