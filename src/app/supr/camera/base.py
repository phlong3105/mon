#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all cameras."""

from __future__ import annotations

__all__ = [
    "Camera",
]

import uuid
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import mon


# region Camera

class Camera(ABC):
    """The base class for all cameras.

    Args:
        id_: Camera's unique ID.
        root: Root directory that stores the data and results.
        subset: Data subset.
        name: Camera name = data name.
        image_loader: Image loader to load the input images.
        image_writer: Image writer to write the processing images.
        save_image: If True, save processing images. Defaults to False.
        save_video: If True, save processing video. Defaults to False.
        save_result : If True, save result. Defaults to False.
        verbose: Verbosity.
    """
    
    def __init__(
        self,
        root        : mon.Path,
        subset      : str,
        name        : str,
        image_loader: Any,
        image_writer: Any,
        id_         : int | str = uuid.uuid4().int,
        save_image  : bool      = False,
        save_video  : bool      = False,
        save_result : bool      = True,
        verbose     : bool      = False,
    ):
        super().__init__()
        if name is None:
            raise ValueError(f"name must be defined.")
        self.id_          = id_
        self.root         = mon.Path(root)
        self.subset       = subset
        self.name         = name
        self.save_image   = save_image
        self.save_video   = save_video
        self.save_result  = save_result
        self.verbose      = verbose
        self.image_loader = image_loader
        self.image_writer = image_writer
        
    @property
    def root(self) -> mon.Path:
        return self._root
    
    @root.setter
    def root(self, root: mon.Path | str | None):
        root = mon.Path(root)
        if root.stem == root.name:
            self._root = root
        else:
            raise ValueError(
                f"root must be a valid directory, but got {self.root}."
            )

    @property
    def subset_dir(self) -> mon.Path:
        if self.subset is None or self.subset == "":
            return self.root
        else:
            return self.root / self.subset
    
    @property
    def result_dir(self) -> mon.Path:
        return self.subset_dir / "result"
    
    @property
    def image_loader(self) -> mon.vision.Loader:
        return self._image_loader
    
    @image_loader.setter
    def image_loader(self, image_loader: Any):
        """Define an image loader object."""
        if isinstance(image_loader, mon.vision.Loader):
            self._image_loader = image_loader
        elif isinstance(image_loader, dict):
            source = mon.Path(image_loader.get("source", None))
            if source.is_dir():
                self._image_loader = mon.ImageLoader(**image_loader)
            else:
                if source.is_basename() or source.is_stem():
                    source = self.subset_dir / f"{source}.mp4"
                if not source.is_video_file():
                    raise ValueError(
                        f"source must be a valid video path, but got {source}."
                    )
                image_loader["source"] = source
                self._image_loader = mon.VideoLoaderCV(**image_loader)
        else:
            raise ValueError(
                f"Cannot initialize image loader with {image_loader}."
            )

    @property
    def image_writer(self) -> mon.vision.Writer | None:
        return self._image_writer
    
    @image_writer.setter
    def image_writer(self, image_writer: Any):
        """Define an image writer object."""
        if not (self.save_image or self.save_video):
            self._image_writer = None
        elif isinstance(image_writer, mon.vision.Writer):
            self._image_writer = image_writer
        elif isinstance(image_writer, dict):
            destination = mon.Path(image_writer.get("destination", None))
            if destination.is_dir():
                self._image_writer = mon.ImageWriter(**image_writer)
            else:
                if destination.is_basename() or destination.is_stem():
                    destination = self.result_dir / f"{destination}.mp4"
                if not destination.is_video_file(exist=False):
                    raise ValueError(
                        f"destination must be a valid video path, but got "
                        f"{destination}."
                    )
                image_writer["destination"] = destination
                image_writer["frame_rate"]  = getattr(self.image_loader, "fps", image_writer["frame_rate"])
                self._image_writer = mon.VideoWriterCV(**image_writer)
        else:
            raise ValueError(
                f"Cannot initialize image writer with {image_writer}."
            )
        
    @abstractmethod
    def on_run_start(self):
        """Called at the beginning of run loop."""
        pass
    
    @abstractmethod
    def run(self):
        """Main run loop."""
        pass
    
    @abstractmethod
    def run_step_end(self, index: int, image: np.ndarray):
        """Perform some postprocessing operations when a run step end."""
        pass
    
    @abstractmethod
    def on_run_end(self):
        """Called at the end of run loop."""
        pass

    @abstractmethod
    def draw(
        self,
        index       : int,
        image       : np.ndarray,
        elapsed_time: float
    ) -> np.ndarray:
        """Visualize the results on the image.

        Args:
            index: Current frame index.
            image: Image to be drawn.
            elapsed_time: Elapsed time per iteration.
        """
        pass

# endregion
