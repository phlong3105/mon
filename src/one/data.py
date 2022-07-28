#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data module implements all helper functions and classes related to storing data
including labels, datasets, dataloaders, and metadata.
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import xmltodict
import yaml
from joblib import delayed
from joblib import Parallel
from matplotlib import pyplot as plt
from munch import Munch
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from yaml import Dumper
from yaml import FullLoader

from one.constants import FILE_HANDLERS
from one.constants import VISION_BACKEND
from one.core import assert_dict
from one.core import assert_dict_contain_key
from one.core import assert_dir
from one.core import assert_image_file
from one.core import assert_json_file
from one.core import assert_list_of
from one.core import assert_sequence_of_length
from one.core import assert_tensor_of_ndim
from one.core import BBoxFormat
from one.core import Callable
from one.core import ComposeTransform
from one.core import console
from one.core import Devices
from one.core import download_bar
from one.core import error_console
from one.core import EvalDataLoaders
from one.core import Ints
from one.core import is_image_file
from one.core import is_same_length
from one.core import is_txt_file
from one.core import is_xml_file
from one.core import ModelPhase_
from one.core import Path_
from one.core import Paths_
from one.core import print_table
from one.core import progress_bar
from one.core import to_list
from one.core import TrainDataLoaders
from one.core import Transforms_
from one.core import VisionBackend
from one.core import VisionBackend_


# H1: - Label ------------------------------------------------------------------

def majority_voting(labels: list[dict]) -> dict:
    """
    It counts the number of appearance of each label, and returns the label with
    the highest count.
    
    Args:
        labels (list[dict]): List of object's label.
    
    Returns:
        A dictionary of the label that has the most votes.
    """
    # Count number of appearance of each label.
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
    
    # Get k (label's id) with max v
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]


class ClassLabel:
    """
    ClassLabel is a list of all classes' dictionaries in the dataset.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/ClassLabel

    Attributes:
        classes (list[dict]):
            List of all classes in the dataset.
    """

    def __init__(self, classes: list[dict]):
        assert_list_of(classes, item_type=dict)
        self._classes = classes

    @staticmethod
    def from_dict(d: dict) -> ClassLabel:
        """
        It takes a dictionary and returns a ClassLabel object.
        
        Args:
            d (dict): dict.
        
        Returns:
            A ClassLabel object.
        """
        assert_dict_contain_key(d, "classes")
        classes = d["classes"]
        classes = Munch.fromDict(classes)
        return ClassLabel(classes=classes)
        
    @staticmethod
    def from_file(path: Path_) -> ClassLabel:
        """
        It creates a ClassLabel object from a `json` file.
        
        Args:
            path (Path_): The path to the `json` file.
        
        Returns:
            A ClassLabel object.
        """
        assert_json_file(path)
        return ClassLabel.from_dict(load_from_file(path))
    
    @staticmethod
    def from_value(value: Any) -> ClassLabel | None:
        """
        It converts an arbitrary value to a ClassLabel.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            A ClassLabel object.
        """
        if isinstance(value, ClassLabel):
            return value
        if isinstance(value, (dict, Munch)):
            return ClassLabel.from_dict(value)
        if isinstance(value, (str, Path)):
            return ClassLabel.from_file(value)
        error_console.log(
            f"`value` must be `ClassLabel`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
        
    @property
    def classes(self) -> list:
        return self._classes

    def color_legend(self, height: int | None = None) -> Tensor:
        """
        It creates a legend of the classes in the dataset.
        
        Args:
            height (int | None): The height of the legend. If None, the legend
                will be 25px high per class.
        
        Returns:
            A tensor of the legend.
        """
        from one.vision.acquisition import to_tensor
        
        num_classes = len(self.classes)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(((num_classes * row_height) + 25, 300, 3), dtype=np.uint8)

        # Loop over the class names + colors
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
        """
        Returns a list of colors for each class in the dataset.
        
        Args:
            key (str): The key to search for. Defaults to id.
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            A list of colors.
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
        """
        
        Returns:
            A dictionary with the label id as the key and the label as the
            value.
        """
        return {label["id"]: label for label in self.classes}

    def ids(
        self,
        key                 : str = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """
        Returns a list of all the ids of the classes in the class list.
        
        Args:
            key (str): The key to search for. Defaults to id.
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            A list of ids.
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
        return self.classes

    @property
    def name2label(self) -> dict[str, dict]:
        """
        
        Returns:
            A dictionary with the label name as the key and the label as the
            value.
        """
        return {c["name"]: c for c in self.classes}

    def names(
        self,
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """
        It returns a list of names of the classes in the dataset.
        
        Args:
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            A list of names of the classes.
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
        """
        Count the number of classes in the dataset, excluding the negative and
        max classes if specified.
        
        Args:
            key (str): The key to search for. Defaults to id.
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            The number of classes in the dataset.
        """
        count = 0
        for c in self.classes:
            if hasattr(c, key):
                if (exclude_negative_key and c[key] <  0  ) or \
                   (exclude_max_key      and c[key] >= 255):
                    continue
                count += 1
        return count

    def get_class(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> dict | None:
        """
        Returns the class with the given key and value, or None if no such
        class exists.
        
        Args:
            key (str): The key to search for. Defaults to id.
            value (int | str | None): The value of the key to search for.
                Defaults to None.
        
        Returns:
            A dictionary of the class that matches the key and value.
        """
        for c in self.classes:
            if hasattr(c, key) and (value == c[key]):
                return c
        return None
    
    def get_class_by_name(self, name: str) -> dict | None:
        """
        Returns the class with the given class name, or None if no such class
        exists.
        
        Args:
            name (str): The name of the class you want to get.
        
        Returns:
            A dictionary of the class with the given name.
        """
        return self.get_class(key="name", value=name)
    
    def get_id(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> int | None:
        """
        Returns the id of the class label that matches the given key and value.
        
        Args:
           key (str): The key to search for. Defaults to id.
           value (int | str | None): The value of the key to search for.
                Defaults to None.
        
        Returns:
            The id of the class.
        """
        class_label: dict = self.get_class(key=key, value=value)
        return class_label["id"] if class_label is not None else None
    
    def get_id_by_name(self, name: str) -> int | None:
        """
        Given a class name, return the class id.
        
        Args:
            name (str): The name of the class you want to get the ID of.
        
        Returns:
            The id of the class.
        """
        class_label = self.get_class_by_name(name=name)
        return class_label["id"] if class_label is not None else None
    
    def get_name(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> str | None:
        """
        Get the name of a class given a key and value.
        
        Args:
            key (str): The key to search for. Defaults to id.
            value (int | str | None): The value of the key to search for.
                Defaults to None.
        
        Returns:
            The name of the class.
        """
        c = self.get_class(key=key, value=value)
        return c["name"] if c is not None else None
       
    def show_color_legend(self, height: int | None = None):
        """Show a pretty color lookup legend using OpenCV drawing functions.

        Args:
            height (int | None): Height of the color legend image.
                Defaults to None.
        """
        color_legend = self.color_legend(height=height)
        plt.imshow(color_legend.permute(1, 2, 0))
        plt.title("Color Legend")
        plt.show()
        
    def print(self):
        """
        Print all classes using `rich` package.
        """
        if not (self.classes and len(self.classes) > 0):
            console.log("[yellow]No class is available.")
            return
        
        console.log("[red]Classlabel:")
        print_table(self.classes)


ClassLabel_ = Union[ClassLabel, str, list, dict]


class BBox:
    """
    Bounding box object with (b1, b2, b3, b4, confidence) format.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
    """
    
    def __init__(
        self,
        b1        : float,
        b2        : float,
        b3        : float,
        b4        : float,
        confidence: float      = 1.0,
        id        : int        = uuid.uuid4().int,
        image_id  : int        = -1,
        class_id  : int        = -1,
        format    : BBoxFormat = BBoxFormat.CXCYWH_NORM,
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
     
    @property
    def bbox(self) -> Tensor:
        """
        It returns a tensor containing the image id, class id, bounding box
        coordinates, and confidence.
        
        Returns:
            A tensor of the image_id, class_id, b1, b2, b3, b4, and confidence.
        """
        return torch.Tensor(
            [
                self.image_id, self.class_id,
                self.b1, self.b2, self.b3, self.b4,
                self.confidence,
            ],
            dtype=torch.float32
        )
    
    @property
    def is_normalized(self) -> bool:
        """
        It checks if the values of the four variables are less than or equal
        to 1.0.
        
        Returns:
          A boolean value.
        """
        return all(i <= 1.0 for i in [self.b1, self.b2, self.b3, self.b4])
   

class Image:
    """
    Image object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
    
    Args:
        id (int | str): The id of the image. This can be an integer or a string.
            This attribute is useful for batch processing where you want to keep
            the objects in the correct frame sequence.
        name (str | None): The name of the image. Defaults to None.
        path (Path_ | None): The path to the image file. Defaults to None.
        image (Tensor[*, C, H, W] | None): The image to be loaded.
            Defaults to None.
        load_on_create (bool): If True, the image will be loaded into memory
            when the object is created. Defaults to False.
        keep_in_memory (bool): If True, the image will be loaded into memory
            and kept there. Defaults to False.
        backend (VisionBackend_): The backend to use for image processing.
            Defaults to VISION_BACKEND.
    """
    
    def __init__(
        self,
        id            : int | str      = uuid.uuid4().int,
        name          : str | None     = None,
        path          : Path_ | None   = None,
        image         : Tensor | None  = None,
        load_on_create: bool           = False,
        keep_in_memory: bool           = False,
        backend       : VisionBackend_ = VISION_BACKEND,
    ):
        from one.vision.acquisition import get_image_shape
        
        self.id             = id
        self.image          = None
        self.keep_in_memory = keep_in_memory
        self.backend        = backend
        
        if path is not None:
            path = Path(path)
            assert_image_file(path)
        self.path: Path = path
        
        if name is None:
            name = str(Path(path).name) if is_image_file(path=path) else f"{id}"
        self.name = name
        
        if load_on_create and image is None:
            image = self.load()

        self.shape = get_image_shape(image) if image is not None else None

        if self.keep_in_memory:
            self.image = image
    
    def load(
        self, path: Path_ | None = None, keep_in_memory: bool = False,
    ) -> Tensor:
        """Load image into memory.
        
        Args:
            path (Path_ | None):
                The path to the image file. Defaults to None.
            keep_in_memory (bool):
                If True, the image will be loaded into memory and kept there.
                Defaults to False.
            
        Returns:
            Return image Tensor of shape [1, C, H, W] to caller.
        """
        from one.vision.acquisition import read_image
        from one.vision.acquisition import get_image_shape
    
        self.keep_in_memory = keep_in_memory
        
        if is_image_file(path):
            self.path = Path(path)
        assert_image_file(path=self.path)
        
        image      = read_image(path=self.path, backend=self.backend)
        self.shape = get_image_shape(image=image) if (image is not None) else self.shape
        
        if self.keep_in_memory:
            self.image = image
        
        return image
        
    @property
    def meta(self) -> dict:
        """
        It returns a dictionary of metadata about the object.
        
        Returns:
            A dictionary with the id, name, path, and shape of the image.
        """
        return {
            "id"   : self.id,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }


class Instance:
    """
    Instance (Segmentation) data consists of object's parts (polygons) and a
    single bounding box covering all parts.
    """
    
    def __init__(
        self,
        bbox      : Tensor | Sequence[float],
        polygons  : Tensor | Sequence[float],
        confidence: float      = 1.0,
        id        : int        = uuid.uuid4().int,
        image_id  : int        = -1,
        class_id  : int        = -1,
        format    : BBoxFormat = BBoxFormat.CXCYWH_NORM,
        *args, **kwargs
    ):
        self.id         = id
        self.image_id   = image_id
        self.class_id   = class_id
        self.bbox       = bbox
        self.polygons   = polygons
        self.confidence = confidence
        self.format     = format
        
    def simplify(self, n: int = -1):
        """
        Simplify each polygon to contains only `n` points.
        
        Args:
            n (int): Number of points to keep in each polygon

        Returns:

        """
        pass


class SegmentationMask(Image):
    """
    Segmentation mask is similar to an image but each pixel is encoded with
    the class id instead of RGB values.
    """
    
    @property
    def class_mask(self) -> Tensor:
        pass
    
    @property
    def one_hot_mask(self) -> Tensor:
        pass


class COCOInstance(Instance):
    """
    COCO instance format.
    """
    
    pass
    

class VOCBBox(BBox):
    """
    VOC bounding box object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
    
    Args:
        name (int | str): This is the name of the object that we are trying to
            identify (i.e., class_id).
        truncated (int): Indicates that the bounding box specified for the
            object does not correspond to the full extent of the object.
            For example, if an object is visible partially in the image then
            we set truncated to 1. If the object is fully visible then set
            truncated to 0.
        difficult (int): An object is marked as difficult when the object is
            considered difficult to recognize. If the object is difficult to
            recognize then we set difficult to 1 else set it to 0.
        bndbox (Tensor | Sequence[float]): Axis-aligned rectangle specifying the
            extent of the object visible in the image.
        pose (str): Specify the skewness or orientation of the image.
            Defaults to Unspecified, which means that the image is not skewed.
    """
    
    def __init__(
        self,
        name     : str,
        truncated: int,
        difficult: int,
        bndbox   : Tensor | Sequence[float],
        pose     : str = "Unspecified",
        *args, **kwargs
    ):
        if isinstance(bndbox, Tensor):
            assert_tensor_of_ndim(bndbox, 1)
            bndbox = bndbox.tolist()
        if isinstance(bndbox, (list, tuple)):
            assert_sequence_of_length(bndbox, 4)
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
        
    def convert_name_to_id(self, class_labels: ClassLabel):
        """
        If the class name is a number, then it is the class id.
        Otherwise, the class id is searched from the ClassLabel object.
        
        Args:
            class_labels (ClassLabel): The ClassLabel containing all classes
                in the dataset.
        """
        self.class_id = int(self.name) \
            if self.name.isnumeric() \
            else class_labels.get_id(key="name", value=self.name)


class DetectionLabel(metaclass=ABCMeta):
    """
    Base class for all detection label format.
    """
    
    @property
    @abstractmethod
    def bboxes(self) -> Tensor:
        pass


class InstanceLabel(metaclass=ABCMeta):
    """
    Base class for all instance label format.
    """

    @abstractmethod
    def draw(self) -> Tensor:
        pass
    
    @property
    @abstractmethod
    def polygons(self) -> Tensor:
        pass


class COCOInstanceLabel(DetectionLabel, InstanceLabel):
    
    def __init__(
        self,
        path    : Path_ | None       = None,
        objects : list[COCOInstance] = [],
        image_id: int                = -1,
        
    ):
        super().__init__()
        
    @staticmethod
    def from_file() -> list[COCOInstanceLabel]:
        """
        Parse multiple labels from a .json file.
        
        Returns:

        """
        pass
    
    @property
    def bboxes(self) -> Tensor:
        pass

    @property
    def polygons(self) -> Tensor:
        pass

    def draw(self) -> Tensor:
        pass


class KITTILabel(DetectionLabel):
    """
    """

    @property
    def bboxes(self) -> Tensor:
        pass


class VOCLabel(DetectionLabel):
    """
    VOC label consists of several bounding boxes. VOC YOLO label corresponds to
    one image and one annotation file.
    
    Args:
        folder (str): Folder that contains the images.
        filename (str): Name of the physical file that exists in the folder.
        path (Path_): The absolute path where the image file is present.
        source (dict): Specifies the original location of the file in a
            database. Since we do not use a database, it is set to `Unknown`
            by default.
        size (dict): Specify the width, height, depth of an image. If the image
            is black and white then the depth will be 1. For color images,
            depth will be 3.
        segmented (int): Signify if the images contain annotations that are
            non-linear (irregular) in shape - commonly referred to as polygons.
            Defaults to 0 (linear shape).
        object (dict | list | None): Contains the object details. If you have
            multiple annotations then the object tag with its contents is
            repeated. The components of the object tags are:
            - name (int, str): This is the name of the object that we are
                trying to identify (i.e., class_id).
            - pose (str): Specify the skewness or orientation of the image.
                Defaults to `Unspecified`, which means that the image is not
                skewed.
            - truncated (int): Indicates that the bounding box specified for
                the object does not correspond to the full extent of the object.
                For example, if an object is visible partially in the image
                then we set truncated to 1. If the object is fully visible then
                set truncated to 0.
            - difficult (int): An object is marked as difficult when the object
                is considered difficult to recognize. If the object is
                difficult to recognize then we set difficult to 1 else set it
                to 0.
            - bndbox (dict): Axis-aligned rectangle specifying the extent of
                the object visible in the image.
        class_labels (ClassLabel | None): ClassLabel object. Defaults to None.
    """
    
    def __init__(
        self,
        path        : Path_ | None      = None,
        objects     : list[VOCBBox]     = [],
        image_id    : int               = -1,
        folder      : str               = "",
        filename    : str               = "",
        image_path  : Path_             = "",
        source      : dict              = {"database": "Unknown"},
        size        : dict              = {"width": 0, "height": 0, "depth": 3},
        segmented   : int               = 0,
        class_labels: ClassLabel | None = None,
        *args, **kwargs
    ):
        from one.vision.shape import box_xyxy_to_cxcywh_norm
        
        super().__init__()
        self.path = Path(path) if isinstance(path, (str, Path)) else path
        
        if len(objects) == 0 and (self.path is None or not is_xml_file(self.path)):
            raise ValueError()
        if len(objects) == 0:
            xml_data   = load_from_file(path)
            assert_dict_contain_key(xml_data, "annotation")
            annotation = xml_data["annotation"]
            folder     = annotation.get("folder"   , folder)
            filename   = annotation.get("filename" , filename)
            image_path = annotation.get("path"     , image_path)
            source     = annotation.get("source"   , source)
            size       = annotation.get("size"     , size)
            width      = int(size.get("width",  0))
            height     = int(size.get("height", 0))
            depth      = int(size.get("depth",  0))
            segmented  = annotation.get("segmented", segmented)
            objects    = annotation.get("object"   , objects)
            objects    = [objects] if not isinstance(objects, list) else objects
            
            assert_list_of(objects, dict)
            for i, o in enumerate(object):
                bndbox   = o.get("bndbox")
                box_xyxy = torch.FloatTensor([
                    int(bndbox["xmin"]), int(bndbox["ymin"]),
                    int(bndbox["xmax"]), int(bndbox["ymax"])
                ])
                o["image_id"] = image_id
                o["bndbox"]   = box_xyxy_to_cxcywh_norm(box_xyxy, height, width)
                o["format"]   = BBoxFormat.CXCYWH_NORM
        
        self.folder    = folder
        self.filename  = filename
        self.path      = Path(path)
        self.source    = source
        self.size      = size
        self.width     = int(self.size.get("width",  0))
        self.height    = int(self.size.get("height", 0))
        self.depth     = int(self.size.get("depth",  0))
        self.segmented = segmented
        self.objects   = [VOCBBox(*o) for o in objects]
        
        if isinstance(class_labels, ClassLabel):
            self.convert_names_to_ids(class_labels=class_labels)
    
    def convert_names_to_ids(
        self, class_labels: ClassLabel, parallel: bool = False
    ):
        """
        Convert `name` property in each `object` to class id.
        
        Args:
            class_labels (ClassLabel): The ClassLabel containing all classes
                in the dataset.
            parallel (bool): If True, run parallely. Defaults to False.
        """
        if parallel:
            def run(i):
                self.objects[i].convert_name_to_id(class_labels)
            
            Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
                delayed(run)(i) for i in range(len(self.objects))
            )
        else:
            for o in self.objects:
                o.convert_name_to_id(class_labels=class_labels)

    @property
    def bboxes(self) -> Tensor:
        return torch.stack([o.bbox for o in self.objects], dim=0)


class YOLOLabel(DetectionLabel):
    """
    YOLO label consists of several bounding boxes. One YOLO label corresponds to
    one image and one annotation file.
    """
    
    def __init__(
        self,
        path    : Path_ | None = None,
        objects : list[BBox]   = [],
        image_id: int          = -1,
        *args, **kwargs
    ):
        super().__init__()
        self.image_id = image_id
        self.path     = Path(path) if isinstance(path, (str, Path)) else path
        
        if len(objects) == 0 and (self.path is None or not is_txt_file(self.path)):
            raise ValueError()
        if len(objects) == 0:
            lines = open(self.path, "r").readlines()
            for l in lines:
                d = l.split(" ")
                objects.append(
                    BBox(
                        image_id = image_id,
                        class_id = int(d[0]),
                        b1       = float(d[1]),
                        b2       = float(d[2]),
                        b3       = float(d[3]),
                        b4       = float(d[4]),
                        format   = BBoxFormat.CXCYWH_NORM
                    )
                )
        self.objects = objects
    
    @property
    def bboxes(self) -> Tensor:
        return torch.stack([o.bbox for o in self.objects], dim = 0)


# H1: - Dataset ----------------------------------------------------------------


class Dataset(data.Dataset, metaclass=ABCMeta):
    """
    Base class for making datasets. It is necessary to override the
    `__getitem__` and `__len__` method.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.root    = Path(root)
        self.split   = split
        self.shape   = shape
        self.verbose = verbose
        
        if transform is not None:
            transform = ComposeTransform(transform)
        if target_transform is not None:
            target_transform = ComposeTransform(target_transform)
        if transforms is not None:
            transforms = ComposeTransform(transforms)
        
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        
        """
        has_transforms         = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can be passed "
                "as argument."
            )

        self.transform        = transform
        self.target_transform = target_transform
        
        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        """
        
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index (int): The index of the sample to be retrieved.
        
        Returns:
            Any.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        
        Returns:
            Length of the dataset.
        """
        pass
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        """
        This function is used to print a string representation of the object.
        """
        return ""


class DataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """
    Base class for all datamodules.
    
    Args:
        root (Path_): Root directory of dataset.
        name (str): Dataset's name.
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        batch_size (int): Number of samples in one forward & backward pass.
            Defaults to 1.
        devices (Device): The devices to use. Defaults to 0.
        shuffle (bool): If True, reshuffle the data at every training epoch.
             Defaults to True.
        collate_fn (Callable | None): Collate function used to fused input items
            together when using `batch_size > 1`.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        name            : str,
        shape           : Ints,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        batch_size      : int                = 1,
        devices         : Devices            = 0,
        shuffle         : bool               = True,
        collate_fn      : Callable | None    = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__()
        self.root             = Path(root)
        self.name             = name
        self.shape            = shape
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        self.batch_size       = batch_size
        self.devices          = devices
        self.shuffle          = shuffle
        self.collate_fn       = collate_fn
        self.verbose          = verbose
        self.dataset_kwargs   = kwargs
        self.train            = None
        self.val              = None
        self.test             = None
        self.predict          = None
        self.class_label      = None
       
    @property
    def devices(self) -> list:
        """
        Returns a list of devices.
        """
        return self._devices

    @devices.setter
    def devices(self, devices: Devices):
        self._devices = to_list(devices)
    
    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        if isinstance(self.class_label, ClassLabel):
            return self.class_label.num_classes()
        return 0
    
    @property
    def num_workers(self) -> int:
        """
        Returns number of workers used in the data loading pipeline.
        """
        # Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 4 * len(self.devices)
        # return 4  # os.cpu_count()

    @property
    def train_dataloader(self) -> TrainDataLoaders | None:
        """
        If the train set exists, return a DataLoader object with the train set,
        otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.train:
            return DataLoader(
                dataset            = self.train,
                batch_size         = self.batch_size,
                shuffle            = self.shuffle,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def val_dataloader(self) -> EvalDataLoaders | None:
        """
        If the validation set exists, return a DataLoader object with the
        validation set, otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.val:
            return DataLoader(
                dataset            = self.val,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def test_dataloader(self) -> EvalDataLoaders | None:
        """
        If the test set exists, return a DataLoader object with the  test set,
        otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.test:
            return DataLoader(
                dataset            = self.test,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def predict_dataloader(self) -> Union[EvalDataLoaders, None]:
        """
        If the prediction set exists, return a DataLoader object with the
        prediction set, otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.predict:
            return DataLoader(
                dataset            = self.predict,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None
    
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        pass
    
    @abstractmethod
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        pass

    @abstractmethod
    def load_class_label(self):
        """
        Load ClassLabel.
        """
        pass
        
    def summarize(self):
        """
        It prints a summary table of the datamodule.
        """
        table = Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Desc")
        
        table.add_row("1", "train",        f"{len(self.train)              if self.train is not None else None}")
        table.add_row("2", "val",          f"{len(self.val)                if self.val   is not None else None}")
        table.add_row("3", "test",         f"{len(self.test)               if self.test  is not None else None}")
        table.add_row("4", "class_labels", f"{self.class_label.num_classes if self.class_label is not None else None}")
        table.add_row("5", "batch_size",   f"{self.batch_size}")
        table.add_row("6", "shape",        f"{self.shape}")
        table.add_row("7", "num_workers",  f"{self.num_workers}")
        console.log(table)


# H2: - Unlabeled --------------------------------------------------------------

class UnlabeledDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of data
    samples.
    """
    pass


class UnlabeledImageDataset(UnlabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of images.
    This is mainly used for unsupervised learning tasks.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root        : Path_,
        split       : str,
        shape       : Ints,
        transform   : Transforms_ | None = None,
        transforms  : Transforms_ | None = None,
        cache_data  : bool               = False,
        cache_images: bool               = False,
        backend     : VisionBackend_     = VISION_BACKEND,
        verbose     : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root       = root,
            split      = split,
            shape      = shape,
            transform  = transform,
            transforms = transforms,
            verbose    = verbose,
            *args, **kwargs
        )
        self.backend = VisionBackend.from_value(backend)
        
        self.images: list[Image] = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.list_images()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
        
        self.filter()
        self.verify()
        if cache_data or not cache_file.is_file():
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
        
    def __getitem__(self, index: int) -> tuple[Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.

        Args:
            index (int): The index of the sample to be retrieved.

        Returns:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Metadata of image.
        """
        item  = self.images[index]
        input = item.image if item.image is not None else item.load()
        meta  = item.meta
        
        if self.transform is not None:
            input, *_ = self.transform(input=input, target=None, dataset=self)
        if self.transforms is not None:
            input, *_ = self.transforms(input=input, target=None, dataset=self)
        return input, meta
        
    def __len__(self) -> int:
        """
        This function returns the length of the images list.
        
        Returns:
            The length of the images list.
        """
        return len(self.images)
        
    @abstractmethod
    def list_images(self):
        """
        List image files.
        """
        pass
    
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass
    
    def verify(self):
        """
        Verify and checking.
        """
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        console.log(f"Number of samples: {len(self.images)}.")
    
    def cache_data(self, path: Path_):
        """
        Cache data to `path`.
        
        Args:
            path (Path_): The path to save the cache.
        """
        cache = {"images": self.images}
        torch.save(cache, str(path))
    
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, meta).
        """
        input, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input, 0)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4.")
        return input, meta


class UnlabeledVideoDataset(UnlabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of video.
    This is mainly used for unsupervised learning tasks.
    """
    pass


class ImageDirectoryDataset(UnlabeledImageDataset):
    """
    A directory of images starting from `root` directory.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root        : Path_,
        split       : str,
        shape       : Ints,
        transform   : Transforms_ | None = None,
        transforms  : Transforms_ | None = None,
        cache_data  : bool               = False,
        cache_images: bool               = False,
        backend     : VisionBackend_     = VISION_BACKEND,
        verbose     : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            shape        = shape,
            transform    = transform,
            transforms   = transforms,
            cache_data   = cache_data,
            cache_images = cache_images,
            backend      = backend,
            verbose      = verbose,
            *args, **kwargs
        )
        
    def list_images(self):
        """
        List image files.
        """
        assert_dir(self.root)
        
        with progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                pattern.rglob("*"),
                description=f"[bright_yellow]Listing {self.split} images"
            ):
                if is_image_file(path):
                    self.images.append(Image(path=path, backend=self.backend))
                    
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass


# H2: - Labeled ----------------------------------------------------------------

class LabeledDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of data
    samples.
    """
    pass


class LabeledImageDataset(LabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of images.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            verbose          = verbose,
            *args, **kwargs
        )
        self.backend     = VisionBackend.from_value(backend)
        self.class_label = ClassLabel.from_value(class_label)
        self.images: list[Image] = []
        if not hasattr(self, "labels"):
            self.labels = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.list_images()
            self.list_labels()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
            self.labels = cache["labels"]
            
        self.filter()
        self.verify()
        if cache_data or not cache_file.is_file():
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
    
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Any, Image]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index (int): The index of the sample to be retrieved.

        Returns:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Target, depending on label type.
            Metadata of image.
        """
        pass
    
    def __len__(self) -> int:
        """
        This function returns the length of the images list.
        
        Returns:
            The length of the images list.
        """
        return len(self.images)
    
    @abstractmethod
    def list_images(self):
        """
        List image files.
        """
        pass

    @abstractmethod
    def list_labels(self):
        """
        List label files.
        """
        pass

    @abstractmethod
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass

    def verify(self):
        """
        Verify and checking.
        """
        if not (is_same_length(self.images, self.labels) and len(self.images) > 0):
            raise RuntimeError(
                f"Number of `images` and `labels` must be the same. "
                f"But got: {len(self.images)} != {len(self.labels)}"
            )
        console.log(f"Number of {self.split} samples: {len(self.images)}.")
        
    def cache_data(self, path: Path_):
        """
        Cache data to `path`.
        
        Args:
            path (Path_): The path to save the cache.
        """
        cache = {
            "images": self.images,
            "labels": self.labels,
        }
        torch.save(cache, str(path))
    
    @abstractmethod
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        pass


class LabeledVideoDataset(LabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of video.
    """
    pass


# H2: - Classification ---------------------------------------------------------

class ImageClassificationDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated classification
    labels stored in a simple JSON format.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[int] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def __getitem__(self, index: int) -> tuple[Tensor, int, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index (int): The index of the sample to be retrieved.

        Returns:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Classification labels.
            Metadata of image.
        """
        item   = self.images[index]
        input  = item.image if item.image is not None else item.load()
        target = self.labels[index]
        meta   = item.meta
        
        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")


class VideoClassificationDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """
    Base type for datasets that represent a collection of videos and a set of
    associated classification labels.
    """
    pass


class ImageClassificationDirectoryTree(ImageClassificationDataset):
    """
    A directory tree whose sub-folders define an image classification dataset.
    """
    
    def list_images(self):
        """
        List image files.
        """
        pass

    def list_labels(self):
        """
        List label files.
        """
        pass
    
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass


# H2: - Detection --------------------------------------------------------------

class ImageDetectionDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of images and a set
    of associated detections.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[DetectionLabel] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index (int): The index of the sample to be retrieved.

        Returns:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Bounding boxes of shape [N, 7].
            Metadata of image.
        """
        item   = self.images[index]
        input  = item.image if item.image is not None else item.load()
        target = self.labels[index].bboxes
        meta   = item.meta

        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input  = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input  = torch.cat(input,  0)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        for i, l in enumerate(target):
            l[:, 0] = i  # add target image index for build_targets()
        return input, target, meta

    
class VideoDetectionDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """
    Base type for datasets that represent a collection of videos and a set of
    associated video detections.
    """
    pass


class COCODetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated object
    detections saved in `COCO Object Detection Format
    <https://cocodataset.org/#format-data>`.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """
        List label files.
        """
        json = self.annotation_file()
        assert_json_file(json)
        json_data = load_from_file(json)
        assert_dict(json_data)
        
        info	    = json_data.get("info", 	   None)
        licenses    = json_data.get("licenses",    None)
        categories  = json_data.get("categories",  None)
        images	    = json_data.get("images",	   None)
        annotations = json_data.get("annotations", None)

        data = Munch()
        temp = Munch()
        for img in images:
            id       = img.get("id",        uuid.uuid4().int)
            filename = img.get("file_name", "")
            image    = Image(
                id   = id,
                name = filename,
                path = "",
            )
            image.coco_url      = img.get("coco_url",      "")
            image.flickr_url    = img.get("flickr_url",    "")
            image.license       = img.get("license",       0 )
            image.date_captured = img.get("date_captured", "")
            data[id]            = image
            temp[filename]      = id
        
    @abstractmethod
    def annotation_file(self) -> Path_:
        """
        Returns the path to json annotation file.
        """
        pass


class VOCDetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated object
    detections saved in `PASCAL VOC format
    <http://host.robots.ox.ac.uk/pascal/VOC>`.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """
        List label files.
        """
        files = self.annotation_files()
        if not (is_same_length(files, self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of `files` and `labels` must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        self.labels: list[VOCLabel] = []
        with progress_bar() as pbar:
            for f in pbar.track(
                files, description=f"[red]Listing {self.split} labels"
            ):
                self.labels.append(
                    VOCLabel(path=f, class_labels=self.class_label)
                )
                
    @abstractmethod
    def annotation_files(self) -> Paths_:
        """
        Returns the path to json annotation files.
        """
        pass


class YOLODetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated object
    detections saved in `YOLO format`.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """
        List label files.
        """
        files = self.annotation_files()
        if not (is_same_length(files, self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of `images` and `labels` must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        self.labels: list[YOLOLabel] = []
        with progress_bar() as pbar:
            for i, f in pbar.track(
                enumerate(files),
                description=f"[red]Listing {self.split} labels"
            ):
                image_id = self.images[i].path.name
                self.labels.append(YOLOLabel(path=f))
        
    @abstractmethod
    def annotation_files(self) -> Paths_:
        """
        Returns the path to json annotation files.
        """
        pass


# H2: - Enhancement ------------------------------------------------------------

class ImageEnhancementDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base type for datasets that represent a collection of images and a set
    of associated enhanced images.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
            index (int): The index of the sample to be retrieved.

        Returns:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Target of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Metadata of image.
        """
        input  = self.images[index].image
        target = self.labels[index].image
        input  = self.images[index].load() if input  is None else input
        target = self.labels[index].load() if target is None else target
        meta   = self.images[index].meta
        
        if self.transform is not None:
            input, *_  = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"[red]Caching {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
    
    def filter(self):
        """
        Filter unwanted samples.
        """
        keep = []
        for i, (img, lab) in enumerate(zip(self.images, self.labels)):
            if is_image_file(img.path) and is_image_file(lab.path):
                keep.append(i)
        self.images = [img for i, img in enumerate(self.images) if i in keep]
        self.labels = [lab for i, lab in enumerate(self.labels) if i in keep]
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        return input, target, meta
    

# H2: - Segmentation -----------------------------------------------------------

class ImageSegmentationDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of images and a set
    of associated semantic segmentations.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[SegmentationMask] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
            index (int): The index of the sample to be retrieved.
          
        Returns:
            input (Tensor[1, C, H, W]): Input sample, optionally transformed by
                the respective transforms.
            target (Tensor[1, C or 1, H, W]): Semantic segmentation mask,
                optionally transformed by the respective transforms.
            meta (Image): Metadata of image.
        """
        input  = self.images[index].image
        target = self.labels[index].image
        input  = self.images[index].load() if input  is None else input
        target = self.labels[index].load() if target is None else target
        meta   = self.images[index].meta

        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
    
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"[red]Caching {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        return input, target, meta


# H2: - Multitask --------------------------------------------------------------

class ImageLabelsDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of images and a set
    of associated multitask predictions.
    """
    pass


class VideoLabelsDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of videos and a set
    of associated multitask predictions.
    """
    pass


# H1: - Serialization ----------------------------------------------------------

def dump_to_file(
    obj        : Any,
    path       : Path_,
    file_format: str | None = None,
    **kwargs
) -> bool | str:
    """
    It dumps an object to a file or a file-like object.
    
    Args:
        obj (Any): The object to be dumped.
        path (Path_): The path to the file to be written.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        A boolean or a string.
    """
    path = Path(path)
    if file_format is None:
        file_format = path.suffix
    assert_dict_contain_key(FILE_HANDLERS, file_format)
    
    handler: BaseFileHandler = FILE_HANDLERS.build(name=file_format)
    if path is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(path, str):
        handler.dump_to_file(obj, path, **kwargs)
    elif hasattr(path, "write"):
        handler.dump_to_fileobj(obj, path, **kwargs)
    else:
        raise TypeError("`path` must be a filename str or a file-object.")


def load_config(cfg: Path_ | dict | Munch) -> Munch:
    """
    Load dictionary from file and convert to namespace using Munch.

    Args:
        cfg (Path_ | dict | Munch): Config filepath that contains
            configuration values or the config dict.
    """
    if isinstance(cfg, (Path, str)):
        d = load_from_file(path=cfg)
    elif isinstance(cfg, (dict, Munch)):
        d = cfg
    else:
        raise TypeError(
            f"`cfg` must be a `dict` or a path to config file. But got: {cfg}."
        )
    
    if d is None:
        raise IOError(f"No configuration is found at: {cfg}.")
    
    cfg = Munch.fromDict(d)
    return cfg


def load_from_file(
    path       : Path_,
    file_format: str | None = None,
    **kwargs
) -> str | dict | None:
    """
    Load a file from a filepath or file-object, and return the data in the file.
    
    Args:
        path (Path_): The path to the file to load.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        The data from the file.
    """
    path = Path(path)
    if file_format is None:
        file_format = path.suffix
    assert_dict_contain_key(FILE_HANDLERS, file_format)

    handler: BaseFileHandler = FILE_HANDLERS.build(name=file_format)
    if isinstance(path, str):
        data = handler.load_from_file(path, **kwargs)
    elif hasattr(path, "read"):
        data = handler.load_from_fileobj(path, **kwargs)
    else:
        raise TypeError("`file` must be a filepath str or a file-object.")
    return data


def merge_files(
    in_paths   : Paths_,
    out_path   : Path_,
    file_format: str | None = None,
) -> bool | str:
    """
    Reads data from multiple files and writes it to a single file.
    
    Args:
        in_paths (Paths_): The input paths to the files you want to merge.
        out_path (Path_): The path to the output file.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        A boolean or a string.
    """
    in_paths = to_list(in_paths)
    in_paths = [Path(p) for p in in_paths]
    
    # Read data
    data = None
    for p in in_paths:
        d = load_from_file(p)
        if isinstance(d, list):
            data = [] if data is None else data
            data += d
        elif isinstance(d, dict):
            data = {} if data is None else data
            data |= d
        else:
            raise TypeError(
                f"Input value must be a `list` or `dict`. But got: {type(d)}."
            )
    
    # Dump data
    return dump_to_file(obj=data, path=out_path, file_format=file_format)


class BaseFileHandler(metaclass=ABCMeta):
    """
    Base file handler implements the template methods (i.e., skeleton) for
    read and write data from/to different file formats.
    """
    
    @abstractmethod
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        It loads a file from a file object.
        
        Args:
            path (Path_): The path to the file to load.
        """
        pass
        
    @abstractmethod
    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It takes a `self` object, an `obj` object, a `path` object, and a
        `**kwargs` object, and returns nothing.
        
        Args:
            obj: The object to be dumped.
            path (Path_): The path to the file to be read.
        """
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It takes an object and returns a string.
        
        Args:
            obj: The object to be serialized.
        """
        pass

    def load_from_file(
        self, path: Path_, mode: str = "r", **kwargs
    ) -> str | dict | None:
        """
        It loads a file from the given path and returns the contents.
        
        Args:
            path (Path_): The path to the file to load from.
            mode (str): The mode to open the file in. Defaults to "r".
        
        Returns:
            The return type is a string, dictionary, or None.
        """
        with open(path, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_file(self, obj, path: Path_, mode: str = "w", **kwargs):
        """
        It writes the object to a file.
        
        Args:
            obj: The object to be serialized.
            path (Path): The path to the file to write to.
            mode (str): The mode in which the file is opened. Defaults to "w".
        """
        with open(path, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


@FILE_HANDLERS.register(name="json")
class JsonHandler(BaseFileHandler):
    """
    JSON file handler.
    """
    
    @staticmethod
    def set_default(obj):
        """
        If the object is a set, range, numpy array, or numpy generic, convert
        it to a list. Otherwise, raise an error.
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A list of the set, range, ndarray, or generic object.
        """
        if isinstance(obj, (set, range)):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"{type(obj)} is not supported for json dump.")
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        This function loads a json file from a file object and returns a
        string, dictionary, or None.
        
        Args:
            path (Path_): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        return json.load(path)

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It dumps the object to a file object.
        
        Args:
            obj: The object to be serialized.
            path (Path_): The path to the file to write to.
        """
        path = Path(path)
        kwargs.setdefault("default", self.set_default)
        json.dump(obj, path, **kwargs)

    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It takes an object and returns a string representation of that object
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A string.
        """
        kwargs.setdefault("default", self.set_default)
        return json.dumps(obj, **kwargs)


@FILE_HANDLERS.register(name="pickle")
@FILE_HANDLERS.register(name="pkl")
class PickleHandler(BaseFileHandler):
    """
    Pickle file handler.
    """
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        This function loads a pickle file from a file object.
        
        Args:
            path (Path_): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        return pickle.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        Takes a Python object, a path to a file, and a set of keyword arguments,
        and writes the object to the file using the pickle module.
        
        Args:
            obj: The object to be pickled.
            path (Path_): The path to the file to be opened.
        """
        path = Path(path)
        kwargs.setdefault("protocol", 4)
        pickle.dump(obj, path, **kwargs)
        
    def dump_to_str(self, obj, **kwargs) -> bytes:
        """
        It takes an object and returns a string representation of that object.
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A bytes object
        """
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)
        
    def load_from_file(self, path: Path_, **kwargs) -> str | dict | None:
        """
        Loads a file from the file system and returns the contents as a string,
        dictionary, or None.
        
        Args:
            path (Path_): Path: The file to load from.
        
        Returns:
            The return value is a string or a dictionary.
        """
        path = Path(path)
        return super().load_from_file(path, mode="rb", **kwargs)
    
    def dump_to_file(self, obj, path: Path_, **kwargs):
        """
        It dumps the object to a file.
        
        Args:
            obj: The object to be serialized.
            path (Path_): The path to the file to which the object is to be
                dumped.
        """
        path = Path(path)
        super().dump_to_file(obj, path, mode="wb", **kwargs)


@FILE_HANDLERS.register(name="xml")
class XmlHandler(BaseFileHandler):
    """
    XML file handler.
    """
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        It takes a path to a file, reads the file, parses the XML, and returns a
        dictionary.
        
        Args:
            path (Path_): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        doc = xmltodict.parse(path.read())
        return doc

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It takes a dictionary, converts it to XML, and writes it to a file.
        
        Args:
            obj: The object to be dumped.
            path (Path_): The path to the file to be read.
        """
        path = Path(path)
        assert_dict(obj)
        with open(path, "w") as path:
            path.write(xmltodict.unparse(obj, pretty=True))
        
    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It takes a dictionary, converts it to XML, and returns the XML as a
        string.
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A string.
        """
        assert_dict(obj)
        return xmltodict.unparse(obj, pretty=True)


@FILE_HANDLERS.register(name="yaml")
@FILE_HANDLERS.register(name="yml")
class YamlHandler(BaseFileHandler):
    """
    YAML file handler.
    """
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        It loads a YAML file from a file object.
        
        Args:
            path (Path): The path to the file to load.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        kwargs.setdefault("Loader", FullLoader)
        return yaml.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It takes a Python object, a path to a file, and a set of keyword
        arguments, and writes the object to the file using the `Dumper` class.
        
        Args:
            obj: The Python object to be serialized.
            path (Path): The file object to dump to.
        """
        path = Path(path)
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, path, **kwargs)

    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It dumps the object to a string.
        
        Args:
            obj: the object to be serialized.
        
        Returns:
            A string.
        """
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)
