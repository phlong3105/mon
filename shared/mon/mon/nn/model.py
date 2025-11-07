#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the ``ModelMixin`` class for managing model attributes
and methods."""

__all__ = [
    "ModelMixin",
]

from typing import Any, Union

import box
import torch

from mon.constants import VERBOSE
from mon.core.console import log
from mon.core.enum import MLType, Task
from mon.core.pathlib import download_url_to_file, Path


# ----- Model -----
class ModelMixin:
    """Mixin for model attributes and methods."""

    arch     : str          = ""         # The model's architecture.
    name     : str          = ""         # The model's name.
    tasks    : list[Task]   = []         # A list of tasks that the model can perform.
    mltypes  : list[MLType] = []         # A list of learning types that the model can perform.
    model_dir: Path         = None       # The model's directory
    zoo      : dict         = box.Box()  # A dictionary containing all pretrained weights of the model.
    
    def parse_weights(
        self,
        weights    : Any,
        num_classes: int  = None,
        overwrite  : bool = False
    ) -> Union[dict, str, int]:
        """Parses and loads pretrained weights for the model.
    
        Args:
            weights: Weights as a ``dict``, ``str``, or ``Path`` to parse;
                Pass ``None`` to skip.
            num_classes: Number of classes for the model. Default: ``None``.
            overwrite: Overwrites existing weights if ``True``. Default: ``False``.
    
        Returns:
            A tuple of (weights, path, num_classes), where:
            - weights: The parsed weights as a ``dict`` or ``None`` if not found.
            - path: The path to the weights file or ``None`` if not applicable.
            - num_classes: The number of classes for the model.
        
        Raises:
            ValueError: If the given weights path is invalid.
        """
        path = None
        
        # Pretrained weights from zoo
        if isinstance(weights, str) and weights in self.zoo:
            url         = self.zoo[weights].get("url",         None)
            path        = self.zoo[weights].get("path",        path)
            num_classes = self.zoo[weights].get("num_classes", num_classes)
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
        """Loads weights into the model.
        
        Args:
            weights: Weights as a ``dict``, ``str``, or ``Path`` to load.
            strict: Whether to strictly enforce that the keys in ``state_dict``
                match the keys returned by this module's ``state_dict`` function.
                Default: ``True``.
            verbose: Whether to print information about the loading process.
                Default: ``True``.
        
        Raises:
            ValueError: If the given weights path is invalid.
        """
        weights, path, _ = self.parse_weights(weights, None)
        if weights:
            self.load_state_dict(weights, strict=strict)
            if verbose:
                log(f"Loaded weights successfully from: {path}.")
