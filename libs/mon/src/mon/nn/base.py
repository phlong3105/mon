#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural network base classes and mixins.

This module implements the base classes and mixins for neural networks.
"""

__all__ = [
    "Container",
    "ModalAdapterMixin",
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
# GLOBAL CONFIGURATIONS (Constants)
# ==============================================================================

# --- Constants (Global defaults, versioning) ---


# --- Environment ---


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---
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
    _zoo      : dict         = box.Box()
    
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
   

class ModalAdapterMixin(RegistrableMixin, ModelZooMixin):
    """A mixin class that provide a unified interface to bridge any model to
    ``mon`` framework.
    """
    pass


# --- Compute Mixins ---
