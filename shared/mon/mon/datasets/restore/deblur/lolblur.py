#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LOL-Blur datasets."""

__all__ = [
    "LOLBlurB",
    "LOLBlurBN",
    "LOLBlurL",
    "LOLBlurLB",
    "LOLBlurLBN",
    "LOLBlurN",
]

import abc

from mon.core import rich
from mon.datasets.core import *


class LOLBlur(VisionDataset, abc.ABC):
    """LOL-Blur dataset."""
    
    name      : str         = "lolblur"
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",      type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName,    type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",        type="image", module=Image,           train=True, test=True),
    }
    classes   : Classes     = None


@DATASETS.register(name="lolblurb")
class LOLBlurB(LOLBlur):
    """LOL-Blur-B (Blur) dataset."""

    tasks: list[Task] = [Task.DEBLUR]

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "b" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images


@DATASETS.register(name="lolblurbn")
class LOLBlurBN(LOLBlur):
    """LOL-Blur-BN (Blur + Noise) dataset."""

    tasks: list[Task] = [Task.DEBLUR, Task.DENOISE]

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "bn" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images


@DATASETS.register(name="lolblurl")
class LOLBlurL(LOLBlur):
    """LOL-Blur-L (Low-Light) dataset."""

    tasks: list[Task] = [Task.LLE]

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "l" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images


@DATASETS.register(name="lolblurlb")
class LOLBlurLB(LOLBlur):
    """LOL-Blur-LB (Low-Light + Blur) dataset."""

    tasks: list[Task] = [Task.DEBLUR, Task.LLE]

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "lb" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images


@DATASETS.register(name="lolblurlbn")
class LOLBlurLBN(LOLBlur):
    """LOL-Blur-LBN (Low-Light + Blur + Noise) dataset."""

    tasks: list[Task] = [Task.DEBLUR, Task.DENOISE, Task.LLE]

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "lbn" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images


@DATASETS.register(name="lolblurn")
class LOLBlurN(LOLBlur):
    """LOL-Blur-N (Noise) dataset."""

    tasks: list[Task] = [Task.DENOISE]

    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "n" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
