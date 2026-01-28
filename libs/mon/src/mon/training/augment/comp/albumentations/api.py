#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""External APIs.

This module collects all external functionalities that are commonly used in
this package. It is intended to be imported by other modules for convenience.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import albumentations as A
# noinspection PyUnusedImports
from albumentations.augmentations import *
# noinspection PyUnusedImports
from albumentations.core.composition import (
    BaseCompose,
    BboxParams,
    KeypointParams,
    OneOf,
    OneOrOther,
    RandomOrder,
    ReplayCompose,
    SelectiveChannelTransform,
    Sequential,
    SomeOf,
)
# noinspection PyUnusedImports
from albumentations.core.serialization import from_dict, load, save, to_dict
# noinspection PyUnusedImports
from albumentations.core.transforms_interface import (
    BasicTransform,
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    Transform3D,
)
# noinspection PyUnusedImports
from albumentations.pytorch import ToTensor3D, ToTensorV2

from mon.core import ALBUMENTATIONS


def __register_transforms(module, prefix: str = ""):
    """Register all transformation classes from the given module and its
    submodules into the ALBUMENTATIONS registry.

    Args:
        module: The module to inspect for transformation classes.
        prefix: The prefix for submodule names. Defaults to "".
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
                        ALBUMENTATIONS.register(name=name, module=obj, replace=True)

            # If it's a package, recursively inspect its submodules
            if is_pkg:
                __register_transforms(sub_module, prefix=module_name + ".")
        except ImportError as e:
            # Skip modules that can't be imported
            continue


__register_transforms(A)
ALBUMENTATIONS.sort()


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
