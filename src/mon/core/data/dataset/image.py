#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Dataset Templates.

This module implements the templates for image-only datasets.
"""

from __future__ import annotations

__all__ = [
    "ImageLoader",
]

import glob

import albumentations as A

from mon.core import pathlib, rich
from mon.core.data import annotation
from mon.core.data.dataset import base
from mon.globals import Split

console             = rich.console
ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
ImageAnnotation     = annotation.ImageAnnotation


# region Image Loader

class ImageLoader(base.MultimodalDataset):
    """A general image loader that retrieves and loads image(s) from a file
    path, file path pattern, or directory.
    """
    
    datapoint_attrs = DatapointAttributes({
        "image": ImageAnnotation,
    })
    
    def __init__(
        self,
        root       : pathlib.Path,
        split      : Split     = Split.PREDICT,
        transform  : A.Compose = None,
        to_tensor  : bool      = False,
        cache_data : bool      = False,
        verbose    : bool      = True,
        *args, **kwargs
    ):
        super().__init__(
            root        = root,
            split		= split,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data	= cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def get_data(self):
        # A single image
        if self.root.is_image_file():
            paths = [self.root]
        # A directory of images
        elif self.root.is_dir() and self.root.exists():
            paths = list(self.root.rglob("*"))
        # A file path pattern
        elif "*" in str(self.root):
            paths = [pathlib.Path(i) for i in glob.glob(str(self.root))]
        else:
            raise IOError(f"Error when listing image files.")
        
        images: list[ImageAnnotation] = []
        with rich.get_progress_bar() as pbar:
            for path in pbar.track(
                sequence    = sorted(paths),
                description = f"[bright_yellow]Listing {self.__class__.__name__} {self.split_str} images"
            ):
                if path.is_image_file():
                    images.append(ImageAnnotation(path=path))
        self.datapoints["image"] = images

# endregion
