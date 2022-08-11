#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data module implements all helper functions and classes related to storing data
including labels, datasets, dataloaders, and metadata.

Taxonomy:
    |
    |__ Label
    |__ Dataset
    |     |__ Unlabeled
    |     |__ Labeled
    |     |__ Classification
    |     |__ Detection
    |     |__ Enhancement
    |     |__ Segmentation
    |     |__ Multitask
    |__ Serialization
"""

from __future__ import annotations

import json
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
from one.core import assert_list
from one.core import assert_list_of
from one.core import assert_number_in_range
from one.core import assert_txt_file
from one.core import assert_xml_file
from one.core import Callable
from one.core import ComposeTransform
from one.core import console
from one.core import Devices
from one.core import download_bar
from one.core import error_console
from one.core import EvalDataLoaders
from one.core import Ints
from one.core import is_image_file
from one.core import is_list_of
from one.core import is_same_length
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

class Label(metaclass=ABCMeta):
    """
    Base class for labels. Label instances represent a logical collection of
    data associated with a particular task for a sample or frame in a dataset.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    
    @classmethod
    @property
    def classname(cls) -> str:
        """
        Returns the name of the class of the object passed to it.
        
        Returns:
            The class name of the object.
        """
        return cls.__name__
    
    @property
    @abstractmethod
    def tensor(self):
        """
        Return the label in tensor format.
        """
        pass


# H2: - Class Label ------------------------------------------------------------

def majority_voting(labels: list[ClassLabel]) -> ClassLabel:
    """
    It counts the number of appearance of each label, and returns the label with
    the highest count.
    
    Args:
        labels (list[ClassLabel]): List of object's classlabels.
    
    Returns:
        A ClassLabel that has the most votes.
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


class ClassLabel(Munch, Label):
    """
    A class label consisting of basic attributes: id, name, and color.
    Each dataset can have additional attributes.
    """

    @property
    def tensor(self):
        """
        Return the label in tensor format.
        """
        return None


class ClassLabels(Label):
    """
    A list of ClassLabels in the dataset.
    
    Args:
        classlabels (list[ClassLabel]): List of all classes in the dataset.
    """

    def __init__(self, classlabels: list[ClassLabel], *args, **kwargs):
        super().__init__()
        assert_list_of(classlabels, item_type=ClassLabel)
        self.classlabels = classlabels

    @classmethod
    def from_list(cls, l: list[dict]) -> ClassLabels:
        """
        It takes a list of dictionary and returns a ClassLabels object.
        
        Args:
            l (list[dict]): List of dictionary.
        
        Returns:
            A ClassLabels object.
        """
        assert_list(l)
        if is_list_of(l, item_type=dict):
            l = [ClassLabel(**c) for c in l]
        return cls(classlabels=l)
    
    @classmethod
    def from_dict(cls, d: dict) -> ClassLabels:
        """
        It takes a dictionary and returns a ClassLabels object.
        
        Args:
            d (dict): dict.
        
        Returns:
            A ClassLabels object.
        """
        assert_dict_contain_key(d, "classlabels")
        return cls.from_list(d["classlabels"])
        
    @classmethod
    def from_file(cls, path: Path_) -> ClassLabels:
        """
        It creates a ClassLabels object from a `json` file.
        
        Args:
            path (Path_): The path to the `json` file.
        
        Returns:
            A ClassLabels object.
        """
        assert_json_file(path)
        return cls.from_dict(load_from_file(path))
    
    @classmethod
    def from_value(cls, value: Any) -> ClassLabels | None:
        """
        It converts an arbitrary value to a ClassLabels.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            A ClassLabels object.
        """
        if isinstance(value, ClassLabels):
            return value
        if isinstance(value, (dict, Munch)):
            return cls.from_dict(value)
        if isinstance(value, list):
            return cls.from_list(value)
        if isinstance(value, (str, Path)):
            return cls.from_file(value)
        error_console.log(
            f"`value` must be `ClassLabels`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
        
    @property
    def classes(self) -> list[ClassLabel]:
        return self.classlabels

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
        return self.classlabels

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
        classlabel: dict = self.get_class(key=key, value=value)
        return classlabel["id"] if classlabel is not None else None
    
    def get_id_by_name(self, name: str) -> int | None:
        """
        Given a class name, return the class id.
        
        Args:
            name (str): The name of the class you want to get the ID of.
        
        Returns:
            The id of the class.
        """
        classlabel = self.get_class_by_name(name=name)
        return classlabel["id"] if classlabel is not None else None
    
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

    @property
    def tensor(self):
        """
        Return the label in tensor format.
        """
        return None
    
    def print(self):
        """
        Print all classes using `rich` package.
        """
        if not (self.classes and len(self.classes) > 0):
            console.log("[yellow]No class is available.")
            return
        
        console.log("Classlabel:")
        print_table(self.classes)


ClassLabels_ = Union[ClassLabels, str, list, dict]


# H2: - Classification ---------------------------------------------------------

class Classification(Label):
    """
    A classification label.
    
    Args:
        id (int): The class id of the classification label. Defaults to -1 means
            unknown.
        label (str): The label string. Defaults to "".
        confidence (float): A confidence in [0.0, 1.0] for the classification.
            Defaults to 1.0.
        logits (Tensor | Sequence | None): Logits associated with the labels.
            Defaults to None.
    """
    
    def __init__(
        self,
        id        : int                      = -1,
        label     : str                      = "",
        confidence: float                    = 1.0,
        logits    : Tensor | Sequence | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.id     = id
        self.label  = label
        self.logits = logits
        
        assert_number_in_range(confidence, 0.0, 1.0)
        self.confidence = confidence

        if self.id == -1.0 and self.label == "":
            raise ValueError(
                f"Either `id` or `name` must be defined. "
                f"But got: {self.id} and {self.label}"
            )

    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.FloatTensor([self.id])
        

class Classifications(Label):
    """
    A list of classifications for an image.

    Args:
        classifications (list[Classification]): A list of Classification
            instances
        logits (Tensor | Sequence): Logits associated with the labels.
    """
    
    def __init__(
        self,
        classifications: list[Classification],
        logits         : Tensor | Sequence,
        *args, **kwargs
    ):
        super().__init__()
        assert_list_of(classifications, Classification)
        self.classifications = classifications
        self.logits          = logits
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.stack([c.tensor for c in self.classifications], dim=0)
    

# H2: - Detection -----------------------------------------------------------

# noinspection PyDefaultArgument
class Detection(Label):
    """
    An object detection.
    
    Args:
        index (int): An index for the object. Defaults to -1.
        id (int): The class id of the detection label. Defaults to -1 means
            unknown.
        label (str): The label string. Defaults to "".
        bbox (Tensor | Sequence[float]): A list of relative bounding box
            coordinates in [0.0, 1.0] in the following format xywh_norm.
        mask (Tensor | None): An instance segmentation mask for the detection
            within its bounding box, which should be a 2D binary list or 0/1
            integer tensor.
        confidence (float): A confidence in [0.0, 1.0] for the detection.
            Defaults to 1.0.
    """
    
    def __init__(
        self,
        index     : int                      = -1,
        id        : int                      = -1,
        label     : str                      = "",
        bbox      : Tensor | Sequence[float] = [],
        mask      : Tensor | None            = [],
        confidence: float                    = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.id    = id
        self.label = label
        self.mask  = mask
        
        assert_number_in_range(confidence, 0.0, 1.0)
        self.confidence = confidence
        
        if not isinstance(bbox, Tensor):
            bbox = torch.FloatTensor(bbox)
        self.bbox = bbox

    @classmethod
    def from_mask(cls, mask: Tensor, label: str, **kwargs):
        """
        Creates a Detection object with its `mask` attribute populated from
        the given full image mask.
        
        The instance mask for the object is extracted by computing the bounding
        rectangle of the non-zero values in the image mask.
        
        Args:
            mask (Tensor): A boolean or 0/1 Tensor.
            label (str): The label string.
            **kwargs: Additional attributes for the `Detection`.
        
        Returns:
            A Detection object.
        """
        pass
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.FloatTensor(
            [
                self.index, self.id,
                self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
                self.confidence,
            ]
        )
        pass
    
    def to_polyline(self, tolerance: int = 2, filled: bool = True) -> Tensor:
        """
        Returns a Polyline representation of this instance. If the detection
        has a mask, the returned polyline will trace the boundary of the mask;
        otherwise, the polyline will trace the bounding box itself.
        
        Args:
            tolerance (int): A tolerance, in pixels, when generating an
                approximate polyline for the instance mask. Typical values are
                1-3 pixels. Defaults to 2.
            filled (Bool): If True, the polyline should be filled.
                Defaults to True.
       
        Returns:
            A Polyline object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : Tensor | None = None,
        frame_size: Ints   | None = None,
        target    : int           = 255
    ) -> Tensor:
        """
        Returns a Segmentation representation of this instance. The detection
        must have an instance mask, i.e., `mask` attribute must be populated.
        You must provide either `mask` or `frame_size` to use this method.
        
        Args:
            mask (Tensor | None): An optional 2D integer numpy array to use as
                an initial mask to which to add this object. Defaults to None.
            frame_size (Ints | None): The shape of the segmentation mask to
                render. This parameter has no effect if a `mask` is provided.
                Defaults to None.
            target (int): The pixel value to use to render the object. If you
                want color mask, just pass in the `id` attribute.
                Defaults to 255.
        
        Returns:
            A Segmentation object.
        """
        pass


# noinspection PyDefaultArgument
class Detections(Label):
    """
    A list of object detections in an image.
    
    Args:
        detections (list[Detection]): A list of Detection objects.
            Defaults to [].
    """
    
    def __init__(
        self,
        detections: list[Detection] = [],
        *args, **kwargs
    ):
        super().__init__()
        assert_list_of(detections, Detection)
        self.detections = detections
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.stack([d.tensor for d in self.detections], dim=0)
    
    def to_polylines(self, tolerance: int = 2, filled: bool = True) -> Tensor:
        """
        Returns a Polylines representation of this instance.
        
        For detections with masks, the returned polylines will trace the
        boundaries of the masks; otherwise, the polylines will trace the
        bounding boxes themselves.
        
        Args:
            tolerance (int): A tolerance, in pixels, when generating an
                approximate polyline for the instance mask. Typical values are
                1-3 pixels. Defaults to 2.
            filled (Bool): If True, the polyline should be filled.
                Defaults to True.
       
        Returns:
            A Polylines object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : Tensor | None = None,
        frame_size: Ints   | None = None,
        target    : int           = 255
    ) -> Tensor:
        """
        Returns a Segmentation representation of this instance.
        
        Only detections with instance masks (i.e., their `mask` attributes
        populated) will be rendered.
        
        Args:
            mask (Tensor | None): An optional 2D integer numpy array to use as
                an initial mask to which to add this object. Defaults to None.
            frame_size (Ints | None): The shape of the segmentation mask to
                render. This parameter has no effect if a `mask` is provided.
                Defaults to None.
            target (int): The pixel value to use to render the object. If you
                want color mask, just pass in the `id` attribute.
                Defaults to 255.
        
        Returns:
            A Segmentation object.
        """
        pass
    

class TemporalDetection(Label):
    """
    A temporal detection in a video whose support is defined by a start and
    end frame.
    """

    @property
    def tensor(self) -> Tensor:
        pass


class COCODetections(Detections):
    """
    """
    pass


class KITTIDetections(Detections):
    """
    """
    pass


# noinspection PyDefaultArgument
class VOCDetections(Detections):
    """
    VOCDetections object consists of several bounding boxes.
    
    One VOCDetections corresponds to one image and one annotation .xml file.
    
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
        classlabels (ClassLabels | None): ClassLabel object. Defaults to None.
    """
    
    def __init__(
        self,
        folder     : str                = "",
        filename   : str                = "",
        path       : Path_              = "",
        source     : dict               = {"database": "Unknown"},
        size       : dict               = {"width": 0, "height": 0, "depth": 3},
        segmented  : int                = 0,
        classlabels: ClassLabels | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.folder      = folder
        self.filename    = filename
        self.path        = Path(path)
        self.source      = source
        self.size        = size
        self.segmented   = segmented
        self.classlabels = classlabels
    
    @classmethod
    def from_file(
        cls, path: Path_, classlabels: ClassLabels | None
    ) -> VOCDetections:
        """
        It creates a VOCDetections object from a .xml file.
        
        Args:
            path (Path_): The path to the .xml file.
            classlabels (ClassLabels | None): ClassLabel object.
                Defaults to None.
            
        Returns:
            A VOCDetections object.
        """
        from one.vision.shape import box_xyxy_to_cxcywh_norm
        
        path = Path(path)
        assert_xml_file(path)
        
        xml_data   = load_from_file(path)
        assert_dict_contain_key(xml_data, "annotation")
       
        annotation = xml_data["annotation"]
        folder     = annotation.get("folder"  , "")
        filename   = annotation.get("filename", "")
        image_path = annotation.get("path"    , "")
        source     = annotation.get("source"  , {"database": "Unknown"})
        size       = annotation.get("size"    , {"width": 0, "height": 0, "depth": 3})
        width      = int(size.get("width" , 0))
        height     = int(size.get("height", 0))
        depth      = int(size.get("depth" , 0))
        segmented  = annotation.get("segmented", 0)
        objects    = annotation.get("object"   , [])
        objects    = [objects] if not isinstance(objects, list) else objects
        
        detections: list[Detection] = []
        for i, o in enumerate(objects):
            name       = o.get["name"]
            bndbox     = o.get["bndbox"]
            bbox       = torch.FloatTensor([bndbox["xmin"], bndbox["ymin"],
                                            bndbox["xmax"], bndbox["ymax"]])
            bbox       = box_xyxy_to_cxcywh_norm(bbox, height, width)
            confidence = o.get("confidence", 1.0)
            truncated  = o.get("truncated" , 0)
            difficult  = o.get("difficult" , 0)
            pose       = o.get("pose", "Unspecified")

            if name.isnumeric():
                id = int(name)
            elif isinstance(classlabels, ClassLabels):
                id = classlabels.get_id(key="name", value=name)
            else:
                id = -1

            detections.append(
                Detection(
                    id         = id,
                    label      = name,
                    bbox       = bbox,
                    confidence = confidence,
                    truncated  = truncated,
                    difficult  = difficult,
                    pose       = pose,
                )
            )
        return cls(
            folder      = folder,
            filename    = filename,
            path        = image_path,
            source      = source,
            size        = size,
            segmented   = segmented,
            detections  = detections,
            classlabels = classlabels
        )


class YOLODetections(Detections):
    """
    YOLO label consists of several bounding boxes. One YOLO label corresponds
    to one image and one annotation file.
    """
    
    @classmethod
    def from_file(cls, path: Path_) -> YOLODetections:
        """
        It creates a YOLODetections object from a .txt file.
        
        Args:
            path (Path_): The path to the .txt file.
        
        Returns:
            A YOLODetections object.
        """
        path = Path(path)
        assert_txt_file(path)
        
        detections: list[Detection] = []
        lines = open(path, "r").readlines()
        for l in lines:
            d          = l.split(" ")
            bbox       = [float(b) for b in d[1:5]]
            confidence = float(d[5]) if len(d) >= 6 else 1.0
            detections.append(
                Detection(
                    id         = int(d[0]),
                    bbox       = bbox,
                    confidence = confidence
                )
            )
        return cls(detections=detections)
        
    
# H2: - Heatmap ----------------------------------------------------------------

class Heatmap(Label):
    """
    A heatmap for an image.
    
    Args:
        map (None): a 2D numpy array.
        range (None): an optional `[min, max]` range of the map's values. If
            None is provided, `[0, 1]` will be assumed if `map` contains
            floating point values, and `[0, 255]` will be assumed if `map`
            contains integer values.
    """

    @property
    def tensor(self) -> Tensor:
        pass


# H2: - Image ------------------------------------------------------------------

class Image(Label):
    """
    Image object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
    
    Args:
        id (int): The id of the image. This can be an integer or a string.
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
        id            : int            = uuid.uuid4().int,
        name          : str    | None  = None,
        path          : Path_  | None  = None,
        image         : Tensor | None  = None,
        load_on_create: bool           = False,
        keep_in_memory: bool           = False,
        backend       : VisionBackend_ = VISION_BACKEND,
        *args, **kwargs
    ):
        from one.vision.acquisition import get_image_shape
        super().__init__(*args, **kwargs)
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
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        if self.keep_in_memory:
            return self.image
        else:
            return self.load()
    

# H2: - Keypoint ---------------------------------------------------------------

# noinspection PyDefaultArgument
class Keypoint(Label):
    """
    A list of keypoints in an image.
    
    Args:
        index (int): An index for the polyline. Defaults to -1.
        id (int): The class id of the polyline label. Defaults to -1 means
            unknown.
        label (str): The label string. Defaults to "".
        points (Tensor | Sequence[float]): A list of lists of `(x, y)` points
            in `[0, 1] x [0, 1]`.
        confidence (float): A confidence in [0.0, 1.0] for the detection.
            Defaults to 1.0.
    """
    
    def __init__(
        self,
        index     : int                      = -1,
        id        : int                      = -1,
        label     : str                      = "",
        points    : Tensor | Sequence[float] = [],
        confidence: float                    = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index  = index
        self.id     = id
        self.label  = label
        
        assert_number_in_range(confidence, 0.0, 1.0)
        self.confidence = confidence
        
        if not isinstance(points, Tensor):
            points = torch.FloatTensor(points)
        self.points = points
       
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        pass


# noinspection PyDefaultArgument
class Keypoints(Label):
    """
    A list of Keypoint objects in an image.
    
    Args:
        keypoints (list[Keypoint]): A list of Keypoint object.
            Defaults to [].
    """
    
    def __init__(
        self,
        keypoints: list[Keypoint] = [],
        *args, **kwargs
    ):
        super().__init__()
        assert_list_of(keypoints, Keypoint)
        self.keypoints = keypoints
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.stack([k.tensor for k in self.keypoints], dim=0)


class COCOKeypoints(Keypoints):
    """
    """
    pass


# H2: - Polyline ---------------------------------------------------------------

# noinspection PyDefaultArgument
class Polyline(Label):
    """
    A set of semantically related polylines or polygons.
    
    Args:
        index (int): An index for the polyline. Defaults to -1.
        id (int): The class id of the polyline label. Defaults to -1 means
            unknown.
        label (str): The label string. Defaults to "".
        points (Tensor | Sequence[float]): A list of lists of `(x, y)` points
            in `[0, 1] x [0, 1]` describing the vertices of each shape in the
            polyline.
        closed (bool): Whether the shapes are closed, i.e., and edge should
            be drawn from the last vertex to the first vertex of each shape.
            Defaults to False.
        filled (bool): Whether the polyline represents polygons, i.e., shapes
            that should be filled when rendering them. Defaults to False.
        confidence (float): A confidence in [0.0, 1.0] for the detection.
            Defaults to 1.0.
        attributes ({}): a dict mapping attribute names to :class:`Attribute`
            instances
    """
    
    def __init__(
        self,
        index     : int                      = -1,
        id        : int                      = -1,
        label     : str                      = "",
        points    : Tensor | Sequence[float] = [],
        closed    : bool                     = False,
        filled    : bool                     = False,
        confidence: float                    = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index  = index
        self.id     = id
        self.label  = label
        self.closed = closed
        self.filled = filled
        
        assert_number_in_range(confidence, 0.0, 1.0)
        self.confidence = confidence
        
        if not isinstance(points, Tensor):
            points = torch.FloatTensor(points)
        self.points = points
       
    @classmethod
    def from_mask(cls, mask: Tensor, label: str, tolerance: int = 2, **kwargs):
        """
        Creates a `Detection` instance with its `mask` attribute populated from
        the given full image mask.
        
        The instance mask for the object is extracted by computing the bounding
        rectangle of the non-zero values in the image mask.
        
        Args:
            mask (Tensor): A boolean or 0/1 Tensor.
            label (str): The label string.
            tolerance (int): A tolerance, in pixels, when generating approximate
                polygons for each region. Typical values are 1-3 pixels.
                Defaults to 2.
            **kwargs: additional attributes for the `Detection`.
        
        Returns:
            A `Detection`.
        """
        pass
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        pass
    
    def to_detection(
        self,
        mask_size : Ints | None = None,
        frame_size: Ints | None = None,
    ) -> Tensor:
        """
        Returns a Detection representation of this instance whose bounding
        box tightly encloses the polyline.
        
        If a `mask_size` is provided, an instance mask of the specified size
        encoding the polyline's shape is included.
       
        Alternatively, if a `frame_size` is provided, the required mask size
        is then computed based off of the polyline points and `frame_size`.
        
        Args:
            mask_size (Ints): An optional shape at which to render an instance
                mask for the polyline.
            frame_size (None): Used when no `mask_size` is provided. An optional
                shape of the frame containing this polyline that is used to
                compute the required `mask_size`.
        
        Returns:
            A Detection object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : Tensor | None = None,
        frame_size: Ints   | None = None,
        target    : int           = 255,
        thickness : int           = 1,
    ) -> Tensor:
        """
        Returns a Segmentation representation of this instance. The detection
        must have an instance mask, i.e., `mask` attribute must be populated.
        You must provide either `mask` or `frame_size` to use this method.
        
        Args:
            mask (Tensor | None): An optional 2D integer numpy array to use as
                an initial mask to which to add this object. Defaults to None.
            frame_size (Ints | None): The shape of the segmentation mask to
                render. This parameter has no effect if a `mask` is provided.
                Defaults to None.
            target (int): The pixel value to use to render the object. If you
                want color mask, just pass in the `id` attribute.
                Defaults to 255.
            thickness (int): The thickness, in pixels, at which to render
                (non-filled) polylines. Defaults to 1.
                
        Returns:
            A Segmentation object.
        """
        pass


# noinspection PyDefaultArgument
class Polylines(Label):
    """
    A list of polylines or polygons in an image.
    
    Args:
        polylines (list[Polyline]): A list of Polyline objects.
            Defaults to [].
    """
    
    def __init__(
        self,
        polylines: list[Polyline] = [],
        *args, **kwargs
    ):
        super().__init__()
        assert_list_of(polylines, Polyline)
        self.polylines = polylines
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.stack([p.tensor for p in self.polylines], dim=0)
    
    def to_detections(
        self,
        mask_size : Ints | None = None,
        frame_size: Ints | None = None,
    ) -> Tensor:
        """
        Returns a Detections representation of this instance whose bounding
        boxes tightly enclose the polylines.
        
        If a `mask_size` is provided, an instance mask of the specified size
        encoding the polyline's shape is included.
       
        Alternatively, if a `frame_size` is provided, the required mask size
        is then computed based off of the polyline points and `frame_size`.
        
        Args:
            mask_size (Ints): An optional shape at which to render an instance
                mask for the polyline.
            frame_size (None): Used when no `mask_size` is provided. An optional
                shape of the frame containing this polyline that is used to
                compute the required `mask_size`.
        
        Returns:
            A Detections object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : Tensor | None = None,
        frame_size: Ints   | None = None,
        target    : int           = 255,
        thickness : int           = 1,
    ) -> Tensor:
        """
        Returns a Segmentation representation of this instance.
        
        You must provide either `mask` or `frame_size` to use this method.
        
        Args:
            mask (Tensor | None): An optional 2D integer numpy array to use as
                an initial mask to which to add this object. Defaults to None.
            frame_size (Ints | None): The shape of the segmentation mask to
                render. This parameter has no effect if a `mask` is provided.
                Defaults to None.
            target (int): The pixel value to use to render the object. If you
                want color mask, just pass in the `id` attribute.
                Defaults to 255.
            thickness (int): The thickness, in pixels, at which to render
                (non-filled) polylines. Defaults to 1.
                
        Returns:
            A Segmentation object.
        """
        pass


# H2: - Regression -------------------------------------------------------------

class Regression(Label):
    """
    A regression value.
    
    Args:
        value (float): The regression value.
        confidence (float): A confidence in [0.0, 1.0] for the classification.
            Defaults to 1.0.
    """
    
    def __init__(
        self,
        value     : float,
        confidence: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.value = value
        
        assert_number_in_range(confidence, 0.0, 1.0)
        self.confidence = confidence
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        return torch.FloatTensor([self.value])
    

# H2: - Segmentation -----------------------------------------------------------

class Segmentation(Label):
    """
    A semantic segmentation for an image.

    Args:
        id (int): The id of the image. This can be an integer or a string.
            This attribute is useful for batch processing where you want to keep
            the objects in the correct frame sequence.
        name (str | None): The name of the image. Defaults to None.
        path (Path_ | None): The path to the image file. Defaults to None.
        mask (Tensor[*, C, H, W] | None): The image with integer values
            encoding the semantic labels. Defaults to None.
        load_on_create (bool): If True, the image will be loaded into memory
            when the object is created. Defaults to False.
        keep_in_memory (bool): If True, the image will be loaded into memory
            and kept there. Defaults to False.
        backend (VisionBackend_): The backend to use for image processing.
            Defaults to VISION_BACKEND.
    """
    
    def __init__(
        self,
        id            : int            = uuid.uuid4().int,
        name          : str    | None  = None,
        path          : Path_  | None  = None,
        mask          : Tensor | None  = None,
        load_on_create: bool           = False,
        keep_in_memory: bool           = False,
        backend       : VisionBackend_ = VISION_BACKEND,
        *args, **kwargs
    ):
        from one.vision.acquisition import get_image_shape
        super().__init__(*args, **kwargs)
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
        
        if load_on_create and mask is None:
            mask = self.load()

        self.shape = get_image_shape(mask) if mask is not None else None

        if self.keep_in_memory:
            self.mask = mask
    
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
        
        mask       = read_image(path=self.path, backend=self.backend)
        self.shape = get_image_shape(image=mask) if (mask is not None) else self.shape
        
        if self.keep_in_memory:
            self.mask = mask
        
        return mask
        
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
    
    @property
    def tensor(self) -> Tensor:
        """
        Return the label in tensor format.
        """
        if self.mask is None:
            self.load()
        return self.mask


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

    @classmethod
    @property
    def classname(cls) -> str:
        """
        Returns the name of the class of the object passed to it.

        Returns:
            The class name of the object.
        """
        return cls.__name__
    

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
        self.classlabels      = None

    @classmethod
    @property
    def classname(cls) -> str:
        """
        Returns the name of the class of the object passed to it.

        Returns:
            The class name of the object.
        """
        return cls.__name__
    
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
        if isinstance(self.classlabels, ClassLabels):
            return self.classlabels.num_classes()
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
            - Build classlabels vocabulary.
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
    def load_classlabels(self):
        """
        Load ClassLabels.
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
        table.add_row("4", "classlabels", f"{self.classlabels.num_classes if self.classlabels is not None else None}")
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
        input = self.images[index].tensor
        meta  = self.images[index].meta
        
        if self.transform is not None:
            input, *_ = self.transform(input=input,  target=None, dataset=self)
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
                description=f"Caching {self.__class__.classname} {self.split} images"
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
            raise ValueError(f"Expect 3 <= `input.ndim` <= 4.")
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
                description=f"[bright_yellow]Listing {self.__class__.classname} "
                            f"{self.split} images"
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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
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
        self.classlabels = ClassLabels.from_value(classlabels)
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
        console.log(f"Cache data to: {path}")
    
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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        self.labels: list[Classifications] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        input  = self.images[index].tensor
        target = self.labels[index].tensor
        meta   = self.images[index].meta
        
        if self.transform is not None:
            input,  *_    = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_    = self.target_transform(input=target, target=None, dataset=self)
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
                description=f"Caching {self.__class__.classname} {self.split} images"
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
            input = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input,  0)
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        target = torch.cat(target, 0)
        return input, target, meta
    

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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_images    : bool                = False,
        cache_data      : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        self.labels: list[Detections] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        input  = self.images[index].tensor
        target = self.labels[index].tensor
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
                description=f"Caching {self.__class__.classname} {self.split} images"
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
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_images    : bool                = False,
        cache_data      : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        json_file = self.annotation_file()
        assert_json_file(json_file)
        json_data = load_from_file(json_file)
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
    <https://host.robots.ox.ac.uk/pascal/VOC>`.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_images    : bool                = False,
        cache_data      : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        
        self.labels: list[VOCDetections] = []
        with progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.classname} {self.split} labels"
            ):
                self.labels.append(
                    VOCDetections.from_file(
                        path        = f,
                        classlabels = self.classlabels
                    )
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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_images    : bool                = False,
        cache_data      : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        
        self.labels: list[YOLODetections] = []
        with progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.classname} {self.split} labels"
            ):
                self.labels.append(YOLODetections.from_file(path=f))
        
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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        input  = self.images[index].tensor
        target = self.labels[index].tensor
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
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.classname} {self.split} labels"
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
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
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
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        self.labels: list[Segmentation] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        input  = self.images[index].tensor
        target = self.labels[index].tensor
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
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.classname} {self.split} labels"
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
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
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

    handler: BaseFileHandler = FILE_HANDLERS.build(name=file_format)
    if isinstance(path, (str, Path)):
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


@FILE_HANDLERS.register(name=".json")
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


@FILE_HANDLERS.register(name=".pickle")
@FILE_HANDLERS.register(name=".pkl")
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


@FILE_HANDLERS.register(name=".xml")
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


@FILE_HANDLERS.register(name=".yaml")
@FILE_HANDLERS.register(name=".yml")
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
