#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimizers.

This package contains various custom optimizers for training neural networks.
"""

from __future__ import annotations

__all__ = []

import importlib
import inspect
import pkgutil

import torch.optim as optim
from torch.optim import Optimizer

from mon.core import OPTIMIZERS


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def __register_optimizers(module, prefix: str = ""):
    """Register all optimizer classes from the given module and its submodules
    into the OPTIMIZERS registry.

    Args:
        module (ModuleType): The module to inspect for optimizer classes.
        prefix (str, optional): The prefix for submodule names. Defaults to "".
    """

    def is_transform_class(obj):
        return (
            inspect.isclass(obj) and
            issubclass(obj, Optimizer) and
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
                        # Add to __all__ and OPTIMIZERS registry __all__.append(name)
                        globals()[name] = obj
                        OPTIMIZERS.register(name=name, module=obj, replace=True)

            # If it's a package, recursively inspect its submodules
            if is_pkg:
                __register_optimizers(sub_module, prefix=module_name + ".")
        except ImportError as e:
            # Skip modules that can't be imported
            continue


__register_optimizers(optim)
del __register_optimizers
OPTIMIZERS.sort()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
