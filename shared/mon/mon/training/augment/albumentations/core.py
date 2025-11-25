#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements core components and integrates ``albumentations`` to
the ``mon`` framework by registering all available transforms.
"""

import importlib
import inspect
import pkgutil

import albumentations as A
from albumentations import *

from mon.core import ALBUMENTATIONS


# ----- Constants -----
TARGET_TYPES = [
    "image",      # The primary input image(s) (e.g., [H, W, C]). Receives geometric, color, and intensity transforms. Uses standard interpolation for geometric transforms.
    "mask",       # Segmentation mask(s) (e.g., [H, W]). Receives geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    "masks",      # Multiple segmentation masks passed together (e.g., [N, H, W]). Processed like mask.
    "bboxes",     # Bounding boxes. Processed according to bbox_params. Requires bbox_params to be set.
    "keypoints",  # Keypoints. Processed according to keypoint_params. Requires keypoint_params to be set.
    "volume",     # A 3D volume (e.g., [D, H, W, C]). Receives 3D geometric transforms, and applicable 2D transforms slice-wise. Color/intensity transforms applied if treated as 'image'.
    "volumes",    # Multiple 3D volumes (e.g., [N, D, H, W, C]). Processed like volume across the first dimension.
    "mask3d",     # A 3D mask (e.g., [D, H, W]). Receives 3D geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    "masks3d"     # Multiple 3D masks (e.g., [N, D, H, W]). Processed like mask3d across the first dimension.
]


# ----- Registry -----
def __register_transforms(module, prefix: str = ""):
    """Recursively inspect a module and its submodules to find transform classes,
    adding them to __all__ and TRANSFORMS registry.
    
    Args:
        module: Module to inspect (e.g., albumentations.augmentations or its submodules).
        prefix: Prefix for module path to track nested module names.
    """
    def is_transform_class(obj):
        return (
            inspect.isclass(obj) and
            issubclass(obj, BasicTransform) and
            obj != BasicTransform and
            not inspect.isabstract(obj)
        )
    
    for _, module_name, is_pkg in pkgutil.walk_packages(module.__path__, prefix=module.__name__ + "."):
        try:
            # Import the submodule
            sub_module = importlib.import_module(module_name)
            
            # Inspect all members of the submodule
            for name, obj in inspect.getmembers(sub_module):
                if is_transform_class(obj) and not name.startswith("_"):
                    # if name not in __all__:
                        # Add to __all__ and TRANSFORMS registry
                        # __all__.append(name)
                        globals()[name] = obj
                        ALBUMENTATIONS.register(name=name, module=obj)
            
            # If it's a package, recursively inspect its submodules
            if is_pkg:
                __register_transforms(sub_module, prefix=module_name + ".")
        except ImportError as e:
            # Skip modules that can't be imported
            continue


# Register all transforms from albumentations.augmentations
__register_transforms(A)
ALBUMENTATIONS.sort()
# print(__all__)
# for k, v in ALBUMENTATIONS.items(): print(f"{k}: {v.__module__}.{v.__name__}")
