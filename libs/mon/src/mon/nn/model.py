#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for model mixin.

This module provides a mixin class that adds metadata and helper methods
to machine learning models, particularly for managing model attributes and
loading pretrained weights.
"""

__all__ = [
    "ModelMetadataMixin",
]

from typing import Any

import box
import torch

from mon.core import download_url_to_file, log, MLType, Path, Task, VERBOSE


# --- Mixins ---
class ModelMetadataMixin:
    """A mixin class that adds metadata and helper methods to a model.
    
    This class defines common dataset attributes for categorization, such as
    supported tasks. This is useful for factory-related operations.
    
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
        _zoo (dict): A dictionary containing all pretrained weights of the model.
            Defaults to an empty dictionary and should be overridden by
            subclasses.
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
        """Getter for the architecture of the model.
        
        Returns:
            str: The architecture of the model.
        """
        return self._arch
    
    @property
    def name(self) -> str:
        """Getter for the name of the model.
        
        Returns:
            str: The name of the model.
        """
        return self._name
    
    @property
    def tasks(self) -> list[Task]:
        """Getter for the tasks that the model can perform.
        
        Returns:
            list[Task]: The tasks that the model can perform.
        """
        return self._tasks
    
    @property
    def mltypes(self) -> list[MLType]:
        """Getter for the learning types that the model can perform.
        
        Returns:
            list[MLType]: The learning types that the model can perform.
        """
        return self._mltypes
    
    @property
    def model_dir(self) -> Path:
        """Getter for the model's directory.
        
        Returns:
            Path: The model's directory.
        """
        return self._model_dir
    
    @property
    def zoo(self) -> dict:
        """Getter for the pretrained weights of the model.
        
        Returns:
            dict: The pretrained weights of the model.
        """
        return self._zoo
    
    # --- Initialize ---
    def parse_weights(
        self,
        weights    : Any,
        num_classes: int  = None,
        overwrite  : bool = False
    ) -> tuple[dict, str, int]:
        """Parses and loads pretrained weights for the model.
    
        Args:
            weights (Any): Weights as a dict, str, or Path to load.
            num_classes (int, optional): The number of classes for the model.
                Defaults to None.
            overwrite (bool): Whether to overwrite existing weights file.
                Defaults to False.
    
        Returns:
            tuple[dict, str, int]: A tuple containing:
                - weights: The parsed weights as a dict or None if not found.
                - path: The path to the weights file or None if not applicable.
                - num_classes: The number of classes for the model.
        
        Raises:
            ValueError: If the given weights path is invalid.
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
        """Loads weights into the model.
        
        Args:
            weights (Any): Weights as a dict, str, or Path to load.
            strict (bool): Whether to strictly enforce that the keys in
                ``state_dict`` match the keys returned by the model's
                ``state_dict()`` function. Defaults to True.
            verbose (bool): Whether to log the loading status. Defaults to True.
        
        Raises:
            NotImplementedError: If the class using ModelMixin does not implement
                ``load_state_dict()``.
        """
        weights, path, _ = self.parse_weights(weights, None)
        if weights:
            if hasattr(self, "load_state_dict"):  # Optional runtime check
                self.load_state_dict(weights, strict=strict)
                if verbose:
                    log(f"Loaded weights successfully from: {path}.")
            else:
                raise NotImplementedError("The class using ModelMixin must implement ``load_state_dict()``.")
