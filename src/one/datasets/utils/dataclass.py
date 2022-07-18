#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""One data classes.
"""

from __future__ import annotations

import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from joblib import delayed
from joblib import Parallel
from matplotlib import pyplot as plt
from munch import Munch
from torch import Tensor

from one.core import BBoxFormat
from one.core import console
from one.core import is_image_file
from one.core import is_json_file
from one.core import is_xml_file
from one.core import load_file
from one.core import print_table
from one.core import VISION_BACKEND
from one.core import VisionBackend
from one.vision.acquisition import read_image
from one.vision.shape import box_xyxy_to_cxcywh_norm
from one.vision.transformation import get_image_shape
from one.vision.transformation import to_tensor


# MARK: - Functional

def majority_voting(labels: list[dict]) -> dict:
    """Get label that has max appearances in the object's labels list."""
    # NOTE: Count number of appearance of each label.
    unique_labels = Munch()
    label_voting  = Munch()
    for label in labels:
        k = label.get("id")
        v = label_voting.get(k)
        if v:
            label_voting[k] = v + 1
        else:
            unique_labels[k] = label
            label_voting[k]  = 1
    
    # NOTE: Get k (label's id) with max v
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]


# MARK: - Modules

class BBox:
    """Bounding box object with (b1, b2, b3, b4, confidence) format.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        b1        : float,
        b2        : float,
        b3        : float,
        b4        : float,
        confidence: float                 = 1.0,
        id        : Union[int, str]       = uuid.uuid4().int,
        image_id  : Union[int, str, None] = None,
        class_id  : Union[int, str, None] = None,
        format    : BBoxFormat            = BBoxFormat.CXCYWH_NORM,
    ):
        self.id         = id
        self.image_id   = image_id
        self.class_id   = class_id
        self.b1         = b1
        self.b2         = b2
        self.b3         = b3
        self.b4         = b4
        self.confidence = confidence
        self.format     = format
    
    # MARK: Property
    
    @property
    def is_normalized(self) -> bool:
        return all(i <= 1.0 for i in [self.b1, self.b2, self.b3, self.b4])
    
    @property
    def label(self) -> Tensor:
        """Return bounding box label.

        Returns:
            box_label (Tensor):
                <image_id>, <class_id>, <x1>, <y1>, <x2>, <y2>, <confidence>
        """
        return torch.Tensor(
            [
                self.image_id,
                self.class_id,
                self.b1, self.b2, self.b3, self.b4,
                self.confidence
            ],
            dtype=torch.float32
        )


class ClassLabel:
    """ClassLabel is a list of all classes' dictionary in the dataset.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/ClassLabel

    Attributes:
        classes (list[dict]):
            List of all classes in the dataset.
    """

    # MARK: Magic Functions

    def __init__(self, classes: list[dict]):
        self._classes = classes

    # MARK: Configure

    @staticmethod
    def create_from_dict(d: dict) -> ClassLabel:
        """Create a `ClassLabel` object from a dictionary.
        
        Args:
            d (dict):
                Dictionary containing all classes.
        """
        if not hasattr(d, "classes"):
            raise FileNotFoundError(f"Given `dict` must contain key `classes`.")
        classes = d["classes"]
        classes = Munch.fromDict(classes)
        return ClassLabel(classes=classes)
        
    @staticmethod
    def create_from_file(path: Union[str, Path]) -> ClassLabel:
        """Create a `ClassLabel` object from a file.
        
        Args:
            path (str, Path):
                Path to file containing all classes.
        """
        if not is_json_file(path=path):
            raise TypeError(f"`path` must be a `json` file. But got: {path}.")
        return ClassLabel.create_from_dict(d=load_file(path=path))
        
    # MARK: Property

    @property
    def classes(self) -> list:
        """Return the list of all classes."""
        return self._classes

    def color_legend(self, height: Union[int, None] = None) -> Tensor:
        """Return a color legend using OpenCV drawing functions.

        References:
            https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/

        Args:
            height (int, None):
                Height of the color legend image. Defaults: `None`.

        Returns:
            legend (Tensor):
                Color legend image.
        """
        num_classes = len(self.classes)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(((num_classes * row_height) + 25, 300, 3), dtype=np.uint8)

        # NOTE: Loop over the class names + colors
        for i, label in enumerate(self.classes):
            color = label.color  # Draw the class name + color on the legend
            color = color[::-1]  # Convert to BGR format since OpenCV operates on BGR format.
            cv2.putText(
                img       = legend,
                text      = label.name,
                org       = (5, (i * row_height) + 17),
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color     = (0, 0, 255),
                thickness = 2
            )
            cv2.rectangle(
                img       = legend,
                pt1       = (150, (i * 25)),
                pt2       = (300, (i * row_height) + 25),
                color     = color,
                thickness = -1
            )
        return to_tensor(image=legend)
        
    def colors(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all classes' colors.
        
        Args:
            key (str):
                Label's key to search from `labels`. Default: `id`.
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0.
                Default: `True`.
            exclude_max_key (bool):
                If `True` only count class's label with key < 255.
                Default: `True`.
        """
        labels_colors = []
        for label in self.classes:
            if hasattr(label, key) and hasattr(label, "color"):
                if (exclude_negative_key and label[key] <  0  ) or \
                   (exclude_max_key      and label[key] >= 255):
                    continue
                labels_colors.append(label.color)

        return labels_colors

    @property
    def id2label(self) -> dict[int, dict]:
        """Return a dictionary of id to class object."""
        return {label["id"]: label for label in self.classes}

    def ids(
        self,
        key                 : str = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all classes' ids at `key`.
        
        Args:
            key (str):
                Key to search from `classes`. Default: `id`.
            exclude_negative_key (bool):
                If `True` only count class with key >= 0. Default: `True`.
            exclude_max_key (bool):
                If `True` only count class with key < 255. Default: `True`.
        """
        ids = []
        for c in self.classes:
            if hasattr(c, key):
                if (exclude_negative_key and c[key] <  0  ) or \
                   (exclude_max_key      and c[key] >= 255):
                    continue
                ids.append(c[key])
        return ids

    @property
    def list(self) -> list:
        """Alias to `classes()`."""
        return self.classes

    @property
    def name2label(self) -> dict[str, dict]:
        """Return a dictionary of {`name`: `class`}."""
        return {c["name"]: c for c in self.classes}

    def names(
        self,
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all classes' names.
        
        Args:
            exclude_negative_key (bool):
                If `True` only count class with key >= 0. Default: `True`.
            exclude_max_key (bool):
                If `True` only count class with key < 255. Default: `True`.
        """
        names = []
        for c in self.classes:
            if hasattr(c, "id"):
                if (exclude_negative_key and c["id"] <  0  ) or \
                   (exclude_max_key      and c["id"] >= 255):
                    continue
                names.append(c["name"])
            else:
                names.append("")
        return names
    
    def num_classes(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> int:
        """Return the number of classes.

        Args:
            key (str):
                Key to search from `classes`. Defaults: `id`.
            exclude_negative_key (bool):
                If `True` only count class with key >= 0. Defaults: `True`.
            exclude_max_key (bool):
                If `True` only count class with key < 255. Defaults: `True`.
        """
        count = 0
        for c in self.classes:
            if hasattr(c, key):
                if (exclude_negative_key and c[key] <  0  ) or \
                   (exclude_max_key      and c[key] >= 255):
                    continue
                count += 1
        return count

    # MARK: Custom Accessors

    def get_class(
        self,
        key  : str                   = "id",
        value: Union[int, str, None] = None
    ) -> Union[dict, None]:
        """Get the class with the given (`key`, `value`) pair.
        
        Args:
            key (str):
                Key to search from `classes`. Defaults: `id`.
            value (int, str, None):
                Key's value to search. Default: `None`.
                
        Returns:
            (dict, None):
                Class dictionary. Default: `None` no class found.
        """
        for c in self.classes:
            if hasattr(c, key) and (value == c[key]):
                return c
        return None
    
    def get_class_by_name(self, name: str) -> Union[dict, None]:
        """Get the class with the given `name`.
        
        Args:
            name (str):
                Name of the class.
                
        Returns:
            (dict, None):
                Class dictionary. Default: `None` no class found.
        """
        return self.get_class(key="name", value=name)
    
    def get_id(
        self,
        key  : str                   = "id",
        value: Union[int, str, None] = None
    ) -> Union[int, None]:
        """Get the id based on the given (`key`, `value`) pair.
        
        Args:
            key (str):
                Key to search from `classes`. Defaults: `id`.
            value (int, str, None):
                Key's value to search. Default: `None`.
                
        Returns:
            (dict, None):
                Class dictionary. Default: `None` no class found.
        """
        class_label: dict = self.get_class(key=key, value=value)
        return class_label["id"] if class_label is not None else None
    
    def get_id_by_name(self, name: str) -> Union[int, None]:
        """Get the id based on the given `name`.
        
        Args:
            name (str):
                Name of the class.
                
        Returns:
            (dict, None):
                Class dictionary. Default: `None` no class found.
        """
        class_label = self.get_class_by_name(name=name)
        return class_label["id"] if class_label is not None else None
    
    def get_name(
        self,
        key  : str                   = "id",
        value: Union[int, str, None] = None
    ) -> Union[str, None]:
        """Get the class's name with the given (`key`, `value`) pair.
        
        Args:
            key (str):
                Key to search from `classes`. Defaults: `id`.
            value (int, str, None):
                Key's value to search. Default: `None`.
                
        Returns:
            (dict, None):
                Class dictionary. Default: `None` no class found.
        """
        c = self.get_class(key=key, value=value)
        return c["name"] if c is not None else None
       
    # MARK: Visualize

    def show_color_legend(self, height: Union[int, None] = None):
        """Show a pretty color lookup legend using OpenCV drawing functions.

        Args:
            height (int, None):
                Height of the color legend image. Default: `None`.
        """
        color_legend = self.color_legend(height=height)
        plt.imshow(color_legend.permute(1, 2, 0))
        plt.title("Color Legend")
        plt.show()
        
    # MARK: Print
    
    def print(self):
        """Print all classes using `rich` package."""
        if not (self.classes and len(self.classes) > 0):
            console.log("[yellow]No class is available.")
            return
        
        console.log("[red]Classlabel:")
        print_table(self.classes)


class Image:
    """Image object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
    
    Args:
        id (int, str):
            Image ID. This attribute is useful for batch processing where you
            want to keep the objects in the correct frame sequence.
        name (str):
            Image name with extension. Default: `None`.
        path (str, Path, None):
            Image path. Default: `None`.
        image (Tensor, None):
            3d array[H, W, C] or Tensor[C, H, W] representing an image.
            Default: `None`.
        load_on_create (bool):
            If `True` attempt to load image into memory (may cause OOM error).
            Default: `False`.
        keep_in_memory (bool):
            If `True` keep the image in memory after loaded (may cause OOM error).
            Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        id            : Union[int, str]           = uuid.uuid4().int,
        name          : Union[str, None]          = None,
        path          : Union[str, Path, None]    = None,
        image         : Union[Tensor, None]       = None,
        load_on_create: bool                      = False,
        keep_in_memory: bool                      = False,
        backend       : Union[VisionBackend, str] = VISION_BACKEND,
    ):
        self.id             = id
        self.image          = None
        self.keep_in_memory = keep_in_memory
        self.backend        = backend
        
        if path is not None:
            if is_image_file(path=path):
                raise FileNotFoundError(f"`path` must be valid. But got: {path}.")
        self.path = path
        
        if name is None:
            name = str(Path(path).name) if is_image_file(path=path) else f"{id}"
        self.name = name
        
        if load_on_create and image is None:
            image = self.load()
        
        self.shape = get_image_shape(image=image) if image is not None else None

        if self.keep_in_memory:
            self.image = image
    
    # MARK: Configure
    
    def load(
        self,
        path          : Union[str, Path, None] = None,
        keep_in_memory: bool                   = False,
    ) -> Tensor:
        """Load image into memory.
        
        Args:
            path (str, Path, None):
                Image path. Default: `None`.
            keep_in_memory (bool):
                If `True` keep the image in memory after loaded (may cause OOM
                error). Default: `False`.
            
        Returns:
            image (Tensor[1, C, H, W]):
                Return image Tensor to caller.
        """
        self.keep_in_memory = keep_in_memory
        
        if is_image_file(path=path):
            self.path = path
        if not is_image_file(path=self.path):
            raise FileNotFoundError(f"`path` must be valid. But got: {self.path}.")
        
        image      = read_image(path=path, backend=self.backend)
        self.shape = get_image_shape(image=image) if (image is not None) else self.shape
        
        if self.keep_in_memory:
            self.image = image
        
        return image
    
    # MARK: Property
    
    @property
    def meta(self) -> dict:
        """Return meta data."""
        return {
            "id"   : self.id,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }


class KITTILabel:
    pass


class VOCBBox(BBox):
    """VOC bounding box object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
    
    Args:
        name (int, str):
            This is the name of the object that we are trying to
            identify (i.e., class_id).
        truncated (int):
            Indicates that the bounding box specified for the object does not
            correspond to the full extent of the object. For example, if an
            object is visible partially in the image then we set truncated to 1.
            If the object is fully visible then set truncated to 0.
        difficult (int):
            An object is marked as difficult when the object is considered
            difficult to recognize. If the object is difficult to recognize
            then we set difficult to 1 else set it to 0.
        bndbox (Tensor, list, tuple):
            Axis-aligned rectangle specifying the extent of the object visible
            in the image.
        pose (str):
            Specify the skewness or orientation of the image.
            Default: `Unspecified`, which means that the image is not skewed.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        name     : str,
        truncated: int,
        difficult: int,
        bndbox   : Union[Tensor, list, tuple],
        pose     : str = "Unspecified",
        *args, **kwargs
    ):
        super().__init__(
            b1 = bndbox[0],
            b2 = bndbox[1],
            b3 = bndbox[2],
            b4 = bndbox[3],
            *args, **kwargs
        )
        self.name      = name
        self.pose      = pose
        self.truncated = truncated
        self.difficult = difficult
    
    # MARK: Configure
    
    def convert_name_to_id(self, class_labels: ClassLabel):
        """Convert `name` property to class id.
        
        Args:
            class_labels (ClassLabel):
                `ClassLabel` object.
        """
        self.class_id = int(self.name) \
            if self.name.isnumeric() \
            else class_labels.get_id(key="name", value=self.name)


class VOCLabel:
    """VOC label.
    
    Args:
        folder (str):
            Folder that contains the images.
        filename (str):
            Name of the physical file that exists in the folder.
        path (str):
            The absolute path where the image file is present.
        source (dict):
            Specifies the original location of the file in a database.
            Since we do not use a database, it is set to `Unknown` by default.
        size (dict):
            Specify the width, height, depth of an image. If the image is
            black and white then the depth will be `1`. For color images, depth
            will be `3`.
        segmented (int):
            Signify if the images contain annotations that are non-linear
            (irregular) in shape - commonly referred to as polygons.
            Default: `0` (linear shape).
        object (dict, list, None):
            Contains the object details. If you have multiple annotations then
            the object tag with its contents is repeated. The components of the
            object tags are:
                - name (int, str):
                    This is the name of the object that we are trying to
                    identify (i.e., class_id).
                - pose (str):
                    Specify the skewness or orientation of the image.
                    Default: `Unspecified`, which means that the image is not
                    skewed.
                - truncated (int):
		            Indicates that the bounding box specified for the object
		            does not correspond to the full extent of the object. For
		            example, if an object is visible partially in the image
		            then we set truncated to 1. If the object is fully visible
		            then set truncated to 0.
                - difficult (int):
                    An object is marked as difficult when the object is
                    considered difficult to recognize. If the object is
                    difficult to recognize then we set difficult to 1 else set
                    it to 0.
                - bndbox (dict):
                    Axis-aligned rectangle specifying the extent of the object
                    visible in the image.
        class_labels (ClassLabel, None):
            `ClassLabel` object. Default: `None`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        folder      : str,
        filename    : str,
        path        : str,
        source      : dict,
        size        : dict,
        segmented   : int,
        object      : Union[dict, list, None],
        class_labels: Union[ClassLabel, None] = None,
        *args, **kwargs
    ):
        self.folder    = folder
        self.filename  = filename
        self.path      = path
        self.source    = source
        self.size      = size
        self.width     = int(self.size.get("width",  0))
        self.height    = int(self.size.get("height", 0))
        self.depth     = int(self.size.get("depth",  0))
        self.segmented = segmented

        if object is None:
            object = []
        else:
            object = [object] if not isinstance(object, dict) else object
        if not all(isinstance(o, dict) for o in object):
            raise TypeError(f"All elements of `object` must be a `dict`.")
            
        for i, o in enumerate(object):
            bndbox   = o.get("bndbox")
            box_xyxy = torch.FloatTensor([
                int(bndbox["xmin"]), int(bndbox["ymin"]),
                int(bndbox["ymin"]), int(bndbox["ymax"])
            ])
            o["bndbox"] = box_xyxy_to_cxcywh_norm(box_xyxy, self.height, self.width)
            o["format"] = BBoxFormat.CXCYWH_NORM
        self.bboxes = [VOCBBox(*b) for b in object]
        
        if isinstance(class_labels, ClassLabel):
            self.convert_names_to_ids(class_labels=class_labels)
        
    # MARK: Configure

    @staticmethod
    def create_from_dict(d: dict, *args, **kwargs) -> VOCLabel:
        """Create a `ClassLabel` object from a dictionary.
        
        Args:
            d (dict):
                Dictionary containing VOC data.
        """
        if not hasattr(d, "annotation"):
            raise ValueError(f"Given `dict` must contain key `annotation`.")
        d = d["annotation"]
        return VOCLabel(
            folder    = d.get("folder"   , ""),
            filename  = d.get("filename" , ""),
            path      = d.get("path"     , ""),
            source    = d.get("source"   , {"database": "Unknown"}),
            size      = d.get("size"     , {"width": 0, "height": 0, "depth": 3}),
            segmented = d.get("segmented", 0),
            object    = d.get("object"   , []),
            *args, **kwargs
        )
        
    @staticmethod
    def create_from_file(path: Union[str, Path], *args, **kwargs) -> VOCLabel:
        """Load VOC label from file.
        
        Args:
            path (str, Path):
                Annotation file.
                
        Returns:
            (VOCLabel):
                `VOCLabel` object.
        """
        if not is_xml_file(path=path):
            raise FileNotFoundError(f"`path` must be a `xml` file. But got: {path}.")
        return VOCLabel.create_from_dict(d=load_file(path=path), *args, **kwargs)
   
    def convert_names_to_ids(
        self,
        class_labels: ClassLabel,
        parallel    : bool = False
    ):
        """Convert `name` property in each `objects` to class id.
        
        Args:
            class_labels (ClassLabel):
                `ClassLabel` object.
            parallel (bool):
                If `True`, run parallely. Default: `False`.
        """
        if parallel:
            def run(i):
                self.bboxes[i].convert_name_to_id(class_labels)
            
            Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
                delayed(run)(i) for i in range(len(self.objects))
            )
        else:
            for o in self.bboxes:
                o.convert_name_to_id(class_labels=class_labels)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
