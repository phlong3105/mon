#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base camera class."""

from __future__ import annotations

__all__ = [
    "Camera",
]

import uuid
from abc import ABC, abstractmethod
from typing import Union

import numpy as np


# region Camera

class Camera(ABC):
    """Base Camera class.

    Attributes:
        id_ (int, str):
            Camera's unique ID.
        dataset (str):
            Dataset name. It is also the name of the directory inside
            `data_dir`. Default: `None`.
        name (str):
            Camera name. It is also the name of the camera's config files.
            Default: `None`.
    """
    
    def __init__(
        self,
        dataset: str,
        name   : str,
        id_    : Union[int, str] = uuid.uuid4().int,
        *args, **kwargs
    ):
        super().__init__()
        if dataset is None:
            raise ValueError(f"`dataset` must be defined.")
        if name is None:
            raise ValueError(f"`name` must be defined.")
        
        self.id_     = id_
        self.dataset = dataset
        self.name    = name

    @abstractmethod
    def run(self):
        """Main run loop."""
        pass

    @abstractmethod
    def run_routine_start(self):
        """Perform operations when run routine starts."""
        pass

    @abstractmethod
    def run_routine_end(self):
        """Perform operations when run routine ends."""
        pass

    @abstractmethod
    def postprocess(self, image: np.ndarray, *args, **kwargs):
        """Perform some postprocessing operations when a run step end.

        Args:
            image (np.ndarray):
                Image.
        """
        pass

    @abstractmethod
    def draw(self, drawing: np.ndarray, elapsed_time: float) -> np.ndarray:
        """Visualize the results on the image.

        Args:
            drawing (np.ndarray):
                Drawing canvas.
            elapsed_time (float):
                Elapsed time per iteration.

        Returns:
            image (np.ndarray):
                Drawn canvas.
        """
        pass

# endregion
