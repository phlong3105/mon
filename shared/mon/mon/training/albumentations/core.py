#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``albumentations.Compose`` class with extended functionalities."""

import importlib
import inspect
import pkgutil

import albumentations as A
from albumentations import *

from mon.core.factory import ALBUMENTATIONS


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
