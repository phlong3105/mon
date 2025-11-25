#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements RealBlurTele dataset for image deblurring."""

__all__ = [
    "RealBlurTeleJ",
    "RealBlurTeleR",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="realblurtelej")
class RealBlurTeleJ(ImageDataset):
    """RealBlurTele-J dataset."""
    
    root_name : str         = "realblurtele"
    tasks     : list[Task]  = [Task.DEBLUR]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "j" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images


@DATASETS.register(name="realblurteler")
class RealBlurTeleR(ImageDataset):
    """RealBlurTele-R dataset."""

    root_name : str         = "realblurtele"
    tasks     : list[Task]  = [Task.DEBLUR]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "r" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path))

        return images
