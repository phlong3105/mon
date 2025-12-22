#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image-based data pools.

This module provides a base class for data pools where one image can have
multiple labels/annotations.
"""

__all__ = [
    "ImageDataPool",
]

from typing import Any

from mon.core import BBoxFormat, create_progress_bar, log, Path
from mon.core.dtypes import bbox as B, Image, Instance
from .base import DataPool
from ..classes import Classes


class ImageDataPool(DataPool):
    """A base class for data pool where one image can have multiple labels/annotations.
    
    This class extends the DataPool base class to handle datasets where each
    image may have multiple associated labels or annotations. It provides
    methods to load, verify, and access the dataset. We assume that the dataset
    contains two main modalities: ``image`` and ``label``.
    
    Attributes:
        image_dir (Path): Absolute path to the image directory.
        label_dir (Path): Absolute path to the label directory.
    """
    
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        classes  : Path | Classes,
        verbose  : bool = True,
        *args, **kwargs
    ):
        """Initializes the ImageDataPool.
        
        Args:
            image_dir (Path): Absolute path to the image directory.
            label_dir (Path): Absolute path to the label directory.
            classes (Path or Classes): Path to classes file or Classes object.
            verbose (bool): If True, enables verbose output. Defaults to True.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        super().__init__(classes=classes, verbose=verbose, *args, **kwargs)
    
    # --- Magic Methods ---
    def __del__(self):
        """Closes the dataset loading mechanism and releases resources."""
        pass
        
    def __len__(self) -> int:
        """Returns the length of the dataset.
        
        Returns:
            int: The number of datapoints in the dataset.
        """
        return len(self._datapoints["image"])
    
    # --- Properties ---
    @property
    def image_dir(self) -> Path:
        """Getter for the image directory.
        
        Returns:
            Path: Path to the image directory.
        """
        return self._image_dir
    
    @image_dir.setter
    def image_dir(self, image_dir: Path):
        """Setter for the image directory.
        
        Args:
            image_dir (Path): Path to the image directory.
            
        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"``image_dir`` directory not found: {image_dir}.")
        self._image_dir = image_dir
    
    @property
    def label_dir(self) -> Path:
        """Getter for the label directory.
        
        Returns:
            Path: Path to the label directory.
        """
        return self._label_dir
    
    @label_dir.setter
    def label_dir(self, label_dir: Path):
        """Setter for the label directory.
        
        Args:
            label_dir (Path): Path to the label directory.
            
        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        label_dir = Path(label_dir)
        if not label_dir.is_dir():
            raise FileNotFoundError(f"``label_dir`` directory not found: {label_dir}.")
        self._label_dir = label_dir
    
    # --- Initialize ---
    def load(self):
        """Initializes and loads all datapoints in the dataset from disk.
        
        After calling this, ``self._datapoints`` will be populated with all
        modalities' data lists. This method can be called internally or externally
        to reload the data if needed.
        """
        # Image
        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(self.image_dir.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=self.image_dir))
        self._datapoints["image"] = images
        
        # Label
        labels: list[list[Instance]] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} label(s)"
            for image in pbar.track(sequence=images, description=desc):
                # Here, we assume that each label file is a YOLO-format .txt file
                # where each line corresponds to one instance/annotation in the
                # image.
                label_file = self.label_dir / f"{image.path.stem}.txt"
                if label_file.is_txt_file(exist=True):
                    labels.append(self._load_label_file(label_file=label_file, image=image))
        self._datapoints["label"] = labels
    
    def _load_label_file(self, label_file: Path, image: Image) -> list[Instance]:
        """Loads all label instances from a label file.
        
        Args:
            label_file (Path): Path to the label file.
            image (Image): The corresponding image object.
        
        Returns:
            list[Instance]: A list of Instance objects representing the labels.
        """
        # For now, we only support loading YOLO bounding boxes
        # Todo: Implement a unified ``load()`` function for ``Instance``
        labels = []
        lines  = B.load(path=label_file, fmt=BBoxFormat.CXCYWHN, imgsz=image.imgsz)
        for l in lines:
            label = Instance(
                data       = lines,
                imgsz      = image.imgsz,
                image_path = image.path,
                root       = self.label_dir,
            )
            labels.append(label)
        return labels
    
    def verify(self):
        """Verifies dataset integrity after loading.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset!")
        
        for k, v in self.datapoints.items():
            if v in [None, []]:
                raise RuntimeError(f"``datapoints`` has no ``{k}`` attributes!")
            elif len(v) != self.__len__():
                raise RuntimeError(f"Number of ``{k}`` items does not match number "
                                   f"of ``image``, got: {len(v)} != {self.__len__()}")
        
        if self.verbose:
            log(f"Number of datapoints: {self.__len__()}.")
    
    # --- Access ---
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Gets a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint.
        """
        datapoint = {}
        for k, v in self._datapoints.items():
            if v is not None and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
            else:
                datapoint[k] = None
        return datapoint
    
    def _get_meta(self, index: int) -> dict[str, Any]:
        """Gets metadata at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the metadata.
        """
        return self._datapoints["image"][index].meta
