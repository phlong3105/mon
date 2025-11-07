#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the FiveK dataset and its variants for image retouching
tasks.
"""

__all__ = [
    "FiveK",
    "FiveKA",
    "FiveKB",
    "FiveKC",
    "FiveKD",
    "FiveKE",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="fivek")
class FiveK(VisionDataset):
    """FiveK dataset."""
    
    root_name : str         = "fivek"
    tasks     : list[Task]  = [Task.RETOUCH, Task.EXPOSURE, Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_c",   type="image", module=Image,           train=True, test=True),
        "ref_a": Modality(name="ref_a",   type="image", module=Image,           train=True, test=True),
        "ref_b": Modality(name="ref_b",   type="image", module=Image,           train=True, test=True),
        "ref_c": Modality(name="ref_c",   type="image", module=Image,           train=True, test=True),
        "ref_d": Modality(name="ref_d",   type="image", module=Image,           train=True, test=True),
        "ref_e": Modality(name="ref_e",   type="image", module=Image,           train=True, test=True),
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
        

@DATASETS.register(name="fiveka")
class FiveKA(VisionDataset):
    """FiveK-A dataset."""
    
    root_name : str         = "fivek"
    tasks     : list[Task]  = [Task.RETOUCH, Task.EXPOSURE, Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_a",   type="image", module=Image,           train=True, test=True),
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
    

@DATASETS.register(name="fivekb")
class FiveKB(FiveKA):
    """FiveK-B dataset."""
    
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_b",   type="image", module=Image,           train=True, test=True),
    }


@DATASETS.register(name="fivekc")
class FiveKC(FiveKA):
    """FiveK-C dataset."""
    
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_c",   type="image", module=Image,           train=True, test=True),
    }
            

@DATASETS.register(name="fivekd")
class FiveKD(FiveKA):
    """FiveK-D dataset."""
    
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_d",   type="image", module=Image,           train=True, test=True),
    }
            

@DATASETS.register(name="fiveke")
class FiveKE(FiveKA):
    """FiveK-E dataset."""
    
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_e",   type="image", module=Image,           train=True, test=True),
    }
