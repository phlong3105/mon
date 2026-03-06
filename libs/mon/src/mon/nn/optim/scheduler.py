#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Schedulers.

This package contains various custom learning rate schedulers for training
neural networks.
"""

from __future__ import annotations

__all__ = []

import importlib
import inspect
import pkgutil

from torch.optim.lr_scheduler import LRScheduler
import torch.optim as optim

from mon.core import SCHEDULERS


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def __register_schedulers(module, prefix: str = ""):
    """Register all scheduler classes from the given module and its submodules
    into the SCHEDULERS registry.

    Args:
        module (ModuleType): The module to inspect for scheduler classes.
        prefix (str, optional): The prefix for submodule names. Defaults to "".
    """

    def is_transform_class(obj):
        return (
            inspect.isclass(obj) and
            issubclass(obj, LRScheduler) and
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
                        # Add to __all__ and SCHEDULERS registry __all__.append(name)
                        globals()[name] = obj
                        SCHEDULERS.register(name=name, module=obj, replace=True)

            # If it's a package, recursively inspect its submodules
            if is_pkg:
                __register_schedulers(sub_module, prefix=module_name + ".")
        except ImportError as e:
            # Skip modules that can't be imported
            continue


__register_schedulers(optim)
del __register_schedulers
SCHEDULERS.sort()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
