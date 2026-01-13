#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for neural networks."""

from __future__ import annotations

__all__ = [
    "Container",
    "ModelAdapterMixin",
    "ModelZooMixin",
    "Module",
    "ModuleDict",
    "ModuleList",
    "ParameterDict",
    "ParameterList",
    "RegistrableMixin",
    "Sequential",
]

from typing import Any

import box
import torch
from torch.nn.modules.container import *
from torch.nn.modules.module import *

from mon.core import download_url_to_file, log, MLType, Path, Task, VERBOSE


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---

class RegistrableMixin:
    """A mixin class that adds metadata attribute to deep learning models for
    factory registration purposes.
    
    Define common model attributes for categorization, such as supported tasks.
    This is useful for factory-related operations.
    
    Attributes:
        _arch (str): The model's architecture. Default is an empty string and
            should be overridden by subclasses.
        _name (str): The model's name. Default is an empty string and should be
            overridden by subclasses.
        _tasks (list[Task]): A list of tasks that the model can perform.
            Defaults to an empty list and should be overridden by subclasses.
        _mltypes (list[MLType]): A list of learning types that the model can
            perform. Defaults to an empty list and should be overridden by
            subclasses.
        _model_dir (Path): The model's directory. Defaults to None and should be
            set by subclasses.
    """
    
    _arch     : str          = ""
    _name     : str          = ""
    _tasks    : list[Task]   = []
    _mltypes  : list[MLType] = []
    _model_dir: Path         = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        arch : str | None        = None,
        name : str | None        = None,
        tasks: list[Task] | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            arch: Architecture of the data container. If provided, it overrides
                class-level default. Defaults to None.
            name: Name of the data container. If provided, it overrides the
                class-level default. Defaults to None.
            tasks: List of supported tasks. If provided, it overrides the
                class-level default. Defaults to None.
        """
        if arch is not None and not isinstance(name, str):
            raise TypeError(f"Expected 'arch' to be a str, but got {type(name).__name__}.")
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Expected 'name' to be a str, but got {type(name).__name__}.")
        if tasks is not None and not isinstance(tasks, list):
            raise TypeError(f"Expected 'tasks' to be a list, but got {type(tasks).__name__}.")

        # If provided, these instance variables will override the class-level defaults
        if arch is not None:
            self._arch = arch
        if name is not None:
            self._name = name
        if tasks is not None:
            # We use list() to create a copy, preventing shared state bugs
            self._tasks = list(tasks)

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance."""
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_arch", "_name", "_tasks"]:
            if attr not in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} must define '{attr}' attribute.")

        # Check for VALID values
        if not isinstance(cls._arch, str):
            raise TypeError(f"Expected '_arch' to be a str, but got {type(cls._arch).__name__}.")
        if not cls._arch:
            raise ValueError(f"Expected '_arch' to be a non-empty str, but got '{cls._arch}'.")

        if not isinstance(cls._name, str):
            raise TypeError(f"Expected '_name' to be a str, but got {type(cls._name).__name__}.")
        if not cls._name:
            raise ValueError(f"Expected '_name' to be a non-empty str, but got '{cls._name}'.")

        if not isinstance(cls._tasks, list):
            raise TypeError(f"Expected '_tasks' to be a list, but got {type(cls._tasks).__name__}.")
        if not cls._tasks:
            raise ValueError(f"Expected '_tasks' to be a non-empty list, but got {cls._tasks}.")

    # --- Properties ---
    @property
    def arch(self) -> str:
        """Return the architecture of the model."""
        return self._arch
    
    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self._name
    
    @property
    def tasks(self) -> list[Task]:
        """Return the tasks that the model can perform."""
        return self._tasks
    
    @property
    def mltypes(self) -> list[MLType]:
        """Return the learning types that the model can perform."""
        return self._mltypes
    
    @property
    def model_dir(self) -> Path:
        """Return the model's directory."""
        return self._model_dir
    

# TODO: Delete later
class ModelZooMixin:
    """A mixin class that adds model zoo functionality to a model.
    
    Attributes:
        _zoo (dict): A dictionary containing all pretrained weights of the model.
            Defaults to an empty dictionary and should be overridden by subclasses.
    """
    
    _zoo: dict = box.Box()
    
    # --- Properties ---
    @property
    def zoo(self) -> dict:
        """Return the pretrained weights of the model."""
        return self._zoo
    
    # --- Initialize ---
    def parse_weights(
        self,
        weights    : Any,
        num_classes: int  = None,
        overwrite  : bool = False
    ) -> tuple[dict, str, int]:
        """Parse and load pretrained weights for the model.
    
        Args:
            weights: Weights as a dict, str, or Path to load.
            num_classes: The number of classes for the model. Defaults to None.
            overwrite: Whether to overwrite an existing weights file. Defaults to False.
    
        Returns:
            A tuple containing:
                - weights: The parsed weights as a dict or None if not found.
                - path: The path to the weights file or None if not applicable.
                - num_classes: The number of classes for the model.
        
        Raises:
            ValueError: If the given ``weights`` path is invalid.
        """
        path = None
        
        # Pretrained weights from zoo
        if isinstance(weights, str) and weights in self._zoo:
            url         = self._zoo[weights].get("url",         None)
            path        = self._zoo[weights].get("path",        path)
            num_classes = self._zoo[weights].get("num_classes", num_classes)
            if url and path and not Path(path).is_weights_file(exist=True):
                download_url_to_file(url, path, overwrite)
        elif isinstance(weights, Path | str):
            path = weights
        
        # Path to weights file
        if path and Path(path).is_weights_file(exist=True):
            weights = torch.load(str(path), weights_only=False)
        
        # State dict
        if isinstance(weights, dict):
            num_classes = weights.get("num_classes", num_classes)
        else:
            weights = None
        
        return weights, path, num_classes
    
    def load_weights(self, weights: Any, strict: bool = True, verbose: bool = VERBOSE):
        """Load weights into the model.
        
        Args:
            weights: Weights as a dict, str, or Path to load.
            strict: Whether to strictly enforce that the keys in
                ``state_dict`` match the keys returned by the model's
                ``state_dict()`` function. Defaults to True.
            verbose: Whether to log the loading status. Defaults to True.
        
        Raises:
            NotImplementedError: If the parent class does not implement ``load_state_dict()``.
        """
        weights, path, _ = self.parse_weights(weights, None)
        
        if weights:
            if hasattr(self, "load_state_dict"):  # Optional runtime check
                self.load_state_dict(weights, strict=strict)
                if verbose:
                    log(f"Loaded weights successfully from: {path}.")
            else:
                raise NotImplementedError("The class using ModelMixin must implement ``load_state_dict()``.")


# TODO: Delete later
class ModelAdapterMixin(RegistrableMixin, ModelZooMixin):
    """A mixin class that provide a unified interface to bridge any model to
    ``mon`` framework.
    """
    pass

# endregion
