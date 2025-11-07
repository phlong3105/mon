#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the GTA5NighttimeFog dataset for nighttime image dehazing.

References:
    - Data: https://github.com/jinyeying/nighttime_dehaze
"""

__all__ = [
    "GTA5NighttimeFog",
]

from mon.core import rich
from ....core import *


# ----- Dataset -----
@DATASETS.register(name="gta5nighttimefog")
class GTA5NighttimeFog(VisionDataset):
    """GTA5NighttimeFog dataset."""
    
    name      : str         = "gta5nighttimefog"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(DepthName,    type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",   type="image", module=Image,           train=True, test=True),
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
