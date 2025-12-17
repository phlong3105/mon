#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for classification probabilities data type.

This module provides a base class for handling classification probabilities with
device management capabilities.
"""

__all__ = [
    "Probs",
    "category_id_to_one_hot",
]

import numpy as np

from .base import BaseTensorOrArray


# ----- Processing -----
def category_id_to_one_hot(class_id: int, num_classes: int) -> np.ndarray:
    """Converts a class ID to a one-hot encoded probability array.
    
    Args:
        class_id (int): Class ID.
        num_classes (int): Total number of classes.
        
    Returns:
        numpy.ndarray: One-hot encoded probability array of shape (num_classes)
            where the index corresponding to ``class_id`` is 1.0 and all other
            indices are 0.0.
            
    Raises:
        ValueError: If ``num_classes`` is not a positive integer.
        ValueError: If ``class_id`` is negative.
        ValueError: If ``class_id`` is out of range for ``num_classes``.
    """
    if num_classes <= 0:
        raise ValueError(f"``num_classes`` must be a positive integer, got {num_classes}.")
    if class_id < 0:
        raise ValueError(f"``class_id`` must be a non-negative integer, got {class_id}.")
    if not (0 <= class_id < num_classes):
        raise ValueError(f"``class_id`` is out of range for ``num_classes`` {num_classes}, got {class_id}.")
    
    probs = np.zeros(num_classes, dtype=np.float32)
    probs[class_id] = 1.0
    return probs


# ----- Probs -----
class Probs(BaseTensorOrArray):
    """A base class for classification probabilities data type.

    This class extends BaseTensorOrArray to handle classification probabilities.
    It provides properties to access top-1 and top-5 class indices, and their
    confidence scores.
    
    Attributes:
        data (numpy.ndarray): Probability vector of shape (``num_classes``).
        _num_classes (int): Total number of classes.
    """
    
    def __init__(self, data: np.ndarray | int, num_classes: int = None):
        """Initializes the Probs instance.
        
        Args:
            data (numpy.ndarray or int): Probability vector as a numpy.ndarray
                of shape (``num_classes``), or an integer representing the class
                ID.
            num_classes (int, optional): Total number of classes. Required if
                ``data`` is provided as an integer. Defaults to None.
        
        Raises:
            ValueError: If ``num_classes`` is provided and is not a positive
                integer.
        """
        # Validate and set num_classes if provided
        if num_classes is not None and num_classes <= 0:
            raise ValueError(f"``num_classes`` must be a positive integer, got {num_classes}.")
        self._num_classes = num_classes
        
        super().__init__(data=data)  # This will call the data setter
        
    # ---- Properties -----
    @property
    def data(self) -> np.ndarray:
        """Getter for the probability vector.
        
        This property is overridden to ensure the returned data is a numpy.ndarray.
        Also, it is needed to override the setter to handle integer class IDs.
        
        Returns:
            numpy.ndarray: Probability data of shape (``num_classes``)
        """
        return self._data
    
    @data.setter
    def data(self, data: np.ndarray | int):
        """Setter for the probability vector.
        
        Args:
            data (numpy.ndarray or int): Probability vector as a numpy.ndarray
                of shape (``num_classes``), or an integer representing the class
                ID. If an integer is provided, it will be converted to a one-hot
                encoded vector.
        
        Raises:
            ValueError: If ``data`` is an integer and ``num_classes`` is not
                provided or is invalid.
            TypeError: If ``data`` is not a numpy.ndarray or int.
        """
        if isinstance(data, int):
            if self.num_classes is None:
                raise ValueError("``num_classes`` must be provided when ``data`` is an integer representing class ID.")
            data = category_id_to_one_hot(class_id=data, num_classes=self.num_classes)
        elif isinstance(data, np.ndarray):
            # Set num_classes if not already set
            if self.num_classes is None:
                self._num_classes = data.shape[0]
            # Validate data shape
            elif data.ndim != 1 or data.shape[0] != self.num_classes:
                raise ValueError(f"``data`` must be a 1D array of shape ({self.num_classes}), got {data.shape}.")
        else:
            raise TypeError(f"``data`` must be a numpy.ndarray or int, got {type(data)}.")
        
        self._data = data
    
    @property
    def num_classes(self) -> int:
        """Getter for the number of classes.
        
        Returns:
            int: The number of classes.
        """
        return self._num_classes
    
    @property
    def top1_idx(self) -> int:
        """Getter for the index of the top-1 class with the highest probability.
        
        Returns:
            int: The index of the class with the highest probability.
        """
        return int(np.argmax(self.data))
    
    @property
    def top5_idxes(self) -> list[int]:
        """Getter for the indices of the top-5 classes with the highest
        probabilities.
        
        Returns:
            list[int]: A list of indices of the top-5 classes with the highest
                probabilities.
        """
        return list(np.argsort(self.data)[-5:][::-1])
    
    @property
    def top1(self) -> float:
        """Getter for the confidence score of the top-1 class.
        
        Returns:
            float: The confidence score of the class with the highest
                probability.
        """
        return self.data[self.top1_idx]
    
    @property
    def top5(self) -> np.ndarray:
        """Getter for the confidence scores of the top-5 classes.
        
        Returns:
            numpy.ndarray: An array of confidence scores for the top-5 classes.
        """
        return self.data[self.top5_idxes]
