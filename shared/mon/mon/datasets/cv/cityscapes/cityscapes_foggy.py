#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Cityscapes-Foggy dataset for dehazing tasks.

References:
	- Data: https://www.cityscapes-dataset.com
"""

__all__ = [
    "CityscapesFoggy",
]

import os

import cv2

from mon.core import pathlib, rich
from .cityscapes import Cityscapes
from ...core import *


@DATASETS.register(name="cityscapes_foggy")
class CityscapesFoggy(Cityscapes):
    """Cityscapes-Foggy dataset for dehazing tasks.

    Args:
        root: Root directory path. Default: ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    
    Raises:
        FileNotFoundError: If ``root``/cityscapes directory does not exist.
    """
    
    tasks      : list[Task]  = [Task.DEHAZE]
    splits     : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities : Modalities  = Modalities({
        "image"    : Image,
        "ref_image": Image,
        "semantic" : SemanticMask,  # gtFine
    })
    has_test_gt: bool        = True
    
    def __init__(self, root: pathlib.Path, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def list_primary_data(self) -> list:
        """Lists foggy images, reference images, and semantic maps."""
        patterns = [self.root / self.split_str / "leftImg8bit_foggy"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
        
        ref_images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} reference image(s)"
            for img in pbar.track(sequence=images, description=desc):
                path = img.path.replace(f"{os.sep}leftImg8bit_foggy{os.sep}", f"{os.sep}leftImg8bit{os.sep}")
                stem = path.stem.split("leftImg8bit")[0]
                path = path.parent / f"{stem}leftImg8bit{path.suffix}"
                ref_images.append(Image(path=path.image_file()))
        
        # Semantic segmentation maps
        semantic: list[SemanticMask] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} semantic maps"
            for img in pbar.track(sequence=ref_images, description=desc):
                path = img.path.replace(f"{os.sep}leftImg8bit{os.sep}", f"{os.sep}gtFine{os.sep}")
                semantic.append(
                    SemanticMask(
                        path  = path.image_file(),
                        root  = img.root,
                        flags = cv2.IMREAD_GRAYSCALE
                    )
                )
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        self.datapoints["semantic"]  = semantic
