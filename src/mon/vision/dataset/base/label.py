#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple label types used in vision tasks and
datasets. We try to support all possible data types: :class:`torch.Tensor`,
:class:`numpy.ndarray`, or :class:`Sequence`, but we prioritize
:class:`torch.Tensor`.
"""

from __future__ import annotations

__all__ = [
    "COCODetectionsLabel", "COCOKeypointsLabel", "ClassificationLabel",
    "ClassificationsLabel", "DetectionLabel", "DetectionsLabel", "HeatmapLabel",
    "ImageLabel", "KITTIDetectionsLabel", "KeypointLabel", "KeypointsLabel",
    "PolylineLabel", "PolylinesLabel", "RegressionLabel", "SegmentationLabel",
    "TemporalDetectionLabel", "VOCDetectionsLabel", "YOLODetectionsLabel",
]

import uuid

import numpy as np
import torch

from mon.foundation import file, pathlib
from mon.vision import geometry, image as mi
from mon.vision.nn import data


# region Classification

class ClassificationLabel(data.Label):
    """A classification label for an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        id_: A class ID of the classification data. Defaults to -1 means
            unknown.
        label: A label string. Defaults to ''.
        confidence: A confidence value for the data. Defaults to 1.0.
        logits: Logits associated with the labels. Defaults to None.
    """
    
    def __init__(
        self,
        id_       : int   = -1,
        label     : str   = "",
        confidence: float = 1.0,
        logits    : np.ndarray | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"conf must be between 0.0 and 1.0, but got {confidence}."
            )
        if id_ <= 0 and label == "":
            raise ValueError(
                f"Either id or name must be defined, but got {id_} and {label}."
            )
        self.id_        = id_
        self.label      = label
        self.confidence = confidence
        self.logits     = np.array(logits) if logits is not None else None
    
    @classmethod
    def from_value(cls, value: ClassificationLabel | dict) -> ClassificationLabel:
        """Create a :class:`ClassificationLabel` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return ClassificationLabel(**value)
        elif isinstance(value, ClassificationLabel):
            return value
        else:
            raise ValueError(
                f"value must be a ClassificationLabel class or a dict, but got "
                f"{type(value)}."
            )
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [self.id_, self.label]
        

class ClassificationsLabel(list[ClassificationLabel], data.Label):
    """A list of classification labels for an image. It is used for multi-labels
    or multi-classes classification tasks.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        seq: A list of :class:`ClassificationLabel` objects.
    """

    def __init__(self, seq: list[ClassificationLabel | dict]):
        super().__init__(ClassificationLabel.from_value(value=i) for i in seq)
    
    def __setitem__(self, index: int, item: ClassificationLabel | dict):
        super().__setitem__(index, ClassificationLabel.from_value(item))
    
    def insert(self, index: int, item: ClassificationLabel | dict):
        super().insert(index, ClassificationLabel.from_value(item))
    
    def append(self, item: ClassificationLabel | dict):
        super().append(ClassificationLabel.from_value(item))
    
    def extend(self, other: list[ClassificationLabel | dict]):
        super().extend([ClassificationLabel.from_value(item) for item in other])
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [i.data for i in self]  

    @property
    def ids(self) -> list[int]:
        return [i.id_ for i in self]  

    @property
    def labels(self) -> list[str]:
        return [i.label for i in self]  

# endregion

# endregion


# region Object Detection

class DetectionLabel(data.Label):
    """An object detection data. Usually, it is represented as a list of
    bounding boxes (for an object with multiple parts created by an occlusion),
    and an instance mask.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        id_: A class ID of the detection data. Defaults to -1 means unknown.
        index: An index for the object. Defaults to -1.
        label: Label string. Defaults to ''.
        confidence: A confidence value for the data. Defaults to 1.0.
        bbox: A bounding box's coordinates.
        mask: Instance segmentation masks for the object within its bounding
            bbox, which should be a binary (0/1) 2D sequence or a binary integer
            tensor. Defaults to None.
    """
    
    def __init__(
        self,
        id_       : int   = -1,
        index     : int   = -1,
        label     : str   = "",
        confidence: float = 1.0,
        bbox      : list  = [],
        mask      : list | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"conf must be between 0.0 and 1.0, but got {confidence}."
            )
        if id_ <= 0 and label == "":
            raise ValueError(
                f"Either id or name must be defined, but got {id_} and {label}."
            )
        self.id_        = id_
        self.index      = index
        self.label      = label
        self.confidence = confidence
        self.bbox       = bbox
        self.mask       = mask if mask is not None else None
        
    @classmethod
    def from_mask(cls, mask: np.ndarray, label: str, **kwargs) -> DetectionLabel:
        """Create a :class:`DetectionLabel` object with its :param:`mask`
        attribute populated from the given full image mask. The instance mask
        for the object is extracted by computing the bounding rectangle of the
        non-zero values in the image mask.
        
        Args:
            mask: A binary (0/1) 2D sequence or a binary integer tensor.
            label: A label string.
            **kwargs: Additional attributes for the :class:`DetectionLabel`.
        
        Return:
            A :class:`DetectionLabel` object.
        """
        raise NotImplementedError(f"This function has not been implemented!")
    
    @classmethod
    def from_value(cls, value: DetectionLabel | dict) -> DetectionLabel:
        """Create a :class:`DetectionLabel` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return DetectionLabel(**value)
        elif isinstance(value, DetectionLabel):
            return value
        else:
            raise ValueError(
                f"value must be a DetectionLabel class or a dict, but got "
                f"{type(value)}."
            )
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
            self.id_, self.label, self.confidence, self.index,
        ]
    
    def to_polyline(
        self,
        tolerance: int  = 2,
        filled   : bool = True
    ) -> PolylineLabel:
        """Return a :class:`PolylineLabel` object of this instance. If the
        detection has a mask, the returned polyline will trace the boundary of
        the mask. Otherwise, the polyline will trace the bounding bbox itself.
        
        Args:
            tolerance: A tolerance, in pixels, when generating an approximate
                polyline for the instance mask. Typical values are 1-3 pixels.
                Defaults to 2.
            filled: If True, the polyline should be filled. Defaults to True.
        
        Return:
            A :class:`PolylineLabel` object.
        """
        raise NotImplementedError(f"This function has not been implemented!")
        
    def to_segmentation(
        self,
        mask      : np.ndarray      | None = None,
        image_size: int | list[int] | None = None,
        target    : int                    = 255
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this instance. The
        detection must have an instance mask, i.e., :param:`mask` attribute must
        be populated. You must give either :param:`mask` or :param:`frame_size`
        to use this method.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            image_size: The size of the segmentation mask to render. This
                parameter has no effect if a :param:`mask` is provided. Defaults
                to None.
            target: The pixel value to use to render the object. If you want
                color mask, just pass in the :param:`id` attribute. Defaults to
                255.
        
        Return:
            A :class:`SegmentationLabel` object.
        """
        raise NotImplementedError(f"This function has not been implemented!")


class DetectionsLabel(list[DetectionLabel], data.Label):
    """A list of object detection labels in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        seq: A list of :class:`DetectionLabel` objects.
    """
    
    def __init__(self, seq: list[DetectionLabel | dict]):
        super().__init__(DetectionLabel.from_value(value=i) for i in seq)
    
    def __setitem__(self, index: int, item: DetectionLabel | dict):
        super().__setitem__(index, DetectionLabel.from_value(item))
    
    def insert(self, index: int, item: DetectionLabel | dict):
        super().insert(index, DetectionLabel.from_value(item))
    
    def append(self, item: DetectionLabel | dict):
        super().append(DetectionLabel.from_value(item))
    
    def extend(self, other: list[DetectionLabel | dict]):
        super().extend([DetectionLabel.from_value(item) for item in other])
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [i.data for i in self]

    @property
    def ids(self) -> list[int]:
        return [i.id_ for i in self]  

    @property
    def labels(self) -> list[str]:
        return [i.label for i in self]  

    @property
    def bboxes(self) -> list:
        return [i.bbox for i in self]
    
    def to_polylines(
        self,
        tolerance: int  = 2,
        filled   : bool = True
    ) -> PolylinesLabel:
        """Return a :class:`PolylinesLabel` object of this instance. For
        detections with masks, the returned polylines will trace the boundaries
        of the masks. Otherwise, the polylines will trace the bounding boxes
        themselves.
        
        Args:
            tolerance: A tolerance, in pixels, when generating an approximate
                polyline for the instance mask. Typical values are 1-3 pixels.
                Defaults to 2.
            filled: If True, the polyline should be filled. Defaults to True.
       
        Return:
            A :class:`PolylinesLabel` object.
        """
        raise NotImplementedError(f"This function has not been implemented!")
    
    def to_segmentation(
        self,
        mask      : np.ndarray      | None = None,
        image_size: int | list[int] | None = None,
        target    : int                    = 255
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this instance. Only
        detections with instance masks (i.e., their :param:`mask` attributes
        populated) will be rendered.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            image_size: The shape of the segmentation mask to render. This
                parameter has no effect if a :param:`mask` is provided. Defaults
                to None.
            target: The pixel value to use to render the object. If you want
                color mask, just pass in the :param:`id` attribute. Defaults to
                255.
        
        Return:
            A :class:`SegmentationLabel` object.
        """
        raise NotImplementedError(f"This function has not been implemented!")


class COCODetectionsLabel(DetectionsLabel):
    """A list of object detection labels in COCO format.
    
    See Also: :class:`DetectionsLabel`.
    """
    pass


class KITTIDetectionsLabel(DetectionsLabel):
    """A list of object detection labels in KITTI format.
    
    See Also: :class:`DetectionsLabel`.
    """
    pass


class VOCDetectionsLabel(DetectionsLabel):
    """A list of object detection labels in VOC format. One VOCDetections
    corresponds to one image and one annotation `.xml` file.
    
    See Also: :class:`DetectionsLabel`.
    
    Args:
        path: Absolute path where the image file is present.
        source: Specifies the original location of the file in a database. Since
            we don't use a database, it is set to 'Unknown' by default.
        size: Specify the width, height, depth of an image. If the image is
            black and white, then the depth will be 1. For color images, depth
            will be 3.
        segmented: Signify if the images contain annotations that are non-linear
            (irregular) in shape—commonly called polygons. Defaults to
            0 (linear shape).
        object: Contains the object details. If you have multiple annotations,
            then the object tag with its contents is repeated. The components of
            the object tags are:
            - name: This is the name of the object that we're trying to
              identify (i.e., class_id).
            - pose: Specify the skewness or orientation of the image. Defaults
              to “Unspecified”, which means that the image isn't skewed.
            - truncated: Indicates that the bounding bbox specified for the
              object doesn't correspond to the full extent of the object. For
              example, if an object is visible partially in the image, then we
              set truncated to 1. If the object is fully visible, then the set
              truncated to 0.
            - difficult: An object is marked as difficult when the object is
              considered difficult to recognize. If the object is difficult to
               recognize, then we set difficult to 1 else set it to 0.
            - bndbox: Axis-aligned rectangle specifying the extent of the object
              visible in the image.
        classlabels: ClassLabel object. Defaults to None.
    """
    
    def __init__(
        self,
        path       : pathlib.Path = "",
        source     : dict = {"database": "Unknown"},
        size       : dict = {"width": 0, "height": 0, "depth": 3},
        segmented  : int  = 0,
        classlabels: data.ClassLabels | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.path        = pathlib.Path(path)
        self.source      = source
        self.size        = size
        self.segmented   = segmented
        self.classlabels = classlabels
    
    @classmethod
    def from_file(
        cls,
        path       : pathlib.Path | str,
        classlabels: data.ClassLabels | None = None
    ) -> VOCDetectionsLabel:
        """Create a :class:`VOCDetections` object from a `.xml` file.
        
        Args:
            path: Path to the `.xml` file.
            classlabels: :class:`ClassLabels` object. Defaults to None.
            
        Return:
            A :class:`VOCDetections` object.
        """
        path = pathlib.Path(path)
        if not path.is_xml_file():
            raise ValueError(
                f"path must be a valid path to an .xml file, but got {path}."
            )
        
        xml_data = file.read_from_file(path=path)
        if "annotation" not in xml_data:
            raise ValueError("xml_data must contain the 'annotation' key.")
       
        annotation = xml_data["annotation"]
        folder     = annotation.get("folder", "")
        filename   = annotation.get("file_name", "")
        image_path = annotation.get("path", "")
        source     = annotation.get("source", {"database": "Unknown"})
        size       = annotation.get("size", {"width": 0, "height": 0, "depth": 3})
        width      = int(size.get("width", 0))
        height     = int(size.get("height", 0))
        depth      = int(size.get("depth", 0))
        segmented  = annotation.get("segmented", 0)
        objects    = annotation.get("object", [])
        objects    = [objects] if not isinstance(objects, list) else objects
        
        detections: list[DetectionLabel] = []
        for i, o in enumerate(objects):
            name       = o.get["name"]
            bndbox     = o.get["bndbox"]
            bbox       = torch.FloatTensor([bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]])
            bbox       = geometry.bbox_xyxy_to_cxcywhn(bbox=bbox, height=height, width=width)
            confidence = o.get("confidence", 1.0)
            truncated  = o.get("truncated", 0)
            difficult  = o.get("difficult" , 0)
            pose       = o.get("pose", "Unspecified")

            if name.isnumeric():
                id = int(name)
            elif isinstance(classlabels, data.ClassLabels):
                id = classlabels.get_id(key="name", value=name)
            else:
                id = -1

            detections.append(
                DetectionLabel(
                    id_       = id,
                    label     = name,
                    bbox      = bbox,
                    confidence= confidence,
                    truncated = truncated,
                    difficult = difficult,
                    pose      = pose,
                )
            )
        return cls(
            path        = image_path,
            source      = source,
            size        = size,
            segmented   = segmented,
            detections  = detections,
            classlabels = classlabels
        )


class YOLODetectionsLabel(DetectionsLabel):
    """A list of object detection labels in YOLO format. YOLO label consists of
    several bounding boxes. One YOLO label corresponds to one image and one
    annotation file.
    
    See Also: :class:`DetectionsLabel`.
    """
    
    @classmethod
    def from_file(cls, path: pathlib.Path) -> YOLODetectionsLabel:
        """Create a :class:`YOLODetectionsLabel` object from a `.txt` file.
        
        Args:
            path: Path to the annotation `.txt` file.
        
        Return:
            A :class:`YOLODetections` object.
        """
        path = pathlib.Path(path)
        if not path.is_txt_file():
            raise ValueError(
                f"path must be a valid path to an .txt file, but got {path}."
            )
        
        detections: list[DetectionLabel] = []
        lines = open(path, "r").readlines()
        for l in lines:
            d          = l.split(" ")
            bbox       = [float(b) for b in d[1:5]]
            confidence = float(d[5]) if len(d) >= 6 else 1.0
            detections.append(
                DetectionLabel(
                    id_        = int(d[0]),
                    bbox       = np.array(bbox),
                    confidence= confidence
                )
            )
        return cls(detections=detections)
        

class TemporalDetectionLabel(data.Label):
    """An object detection label in a video whose support is defined by a start
    and end frame. Usually, it is represented as a list of bounding boxes (for
    an object with multiple parts created by an occlusion), and an instance
    mask.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    """
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Heatmap

class HeatmapLabel(data.Label):
    """A heatmap label in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        map: A 2D numpy array.
        range: An optional [min, max] range of the map's values. If None is
            provided, [0, 1] will be assumed if :param:`map` contains floating
            point values, and [0, 255] will be assumed if :param:`map` contains
            integer values.
    """

    @property
    def data(self) -> list | None:
        """The label's data."""
        raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Image

class ImageLabel(data.Label):
    """A ground-truth image label for an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
    
    Args:
        id_: An ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: A name of the image. Defaults to None.
        path: A path to the image file. Defaults to None.
        image: A ground-truth image to be loaded. Defaults to None.
        load_on_create: If True, the image will be loaded into memory when the
            object is created. Defaults to False.
        keep_in_memory: If True, the image will be loaded into memory and kept
            there. Defaults to False.
    """
    
    to_rgb   : bool = True
    to_tensor: bool = False
    normalize: bool = False
    
    def __init__(
        self,
        id_           : int                 = uuid.uuid4().int,
        name          : str          | None = None,
        path          : pathlib.Path | None = None,
        image         : np.ndarray   | None = None,
        load_on_create: bool                = False,
        keep_in_memory: bool                = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.id_            = id_
        self.image          = None
        self.keep_in_memory = keep_in_memory
        
        if path is not None:
            path = pathlib.Path(path)
            if not path.is_image_file():
                raise ValueError(
                    f"path must be a valid path to an image file, but got {path}."
                )
        self.path = path
        
        if name is None:
            name = str(pathlib.Path(path).name) if path.is_image_file() else f"{id_}"
        self.name = name

        if load_on_create and image is None:
            image = self.load()
        
        self.shape = mi.get_image_shape(image=image) if image is not None else None
       
        if self.keep_in_memory:
            self.image = image
    
    def load(
        self,
        path          : pathlib.Path | None = None,
        keep_in_memory: bool = False,
    ) -> np.ndarray:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Defaults to None.
            keep_in_memory: If True, the image will be loaded into memory and
                kept there. Defaults to False.
            
        Return:
            An image of shape HWC.
        """
        self.keep_in_memory = keep_in_memory
        
        if path is not None:
            path = pathlib.Path(path)
            if path.is_image_file():
                self.path = path
        if not self.path.is_image_file():
            raise ValueError(
                f"path must be a valid path to an image file, but got {self.path}."
            )
        
        image = mi.read_image(
            path      = self.path,
            to_rgb    = self.to_rgb,
            to_tensor = self.to_tensor,
            normalize = self.normalize,
        )
        self.shape = mi.get_image_shape(image=image) if (image is not None) else self.shape
        
        if self.keep_in_memory:
            self.image = image
        return image
        
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "id"   : self.id_,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }
    
    @property
    def data(self) -> np.ndarray | None:
        """The label's data."""
        if self.image is None:
            return self.load()
        else:
            return self.image
       

# endregion


# region Keypoint

class KeypointLabel(data.Label):
    """A list keypoints label for a single object in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        id_: The class ID of the polyline data. Defaults to -1 means unknown.
        index: An index for the polyline. Defaults to -1.
        label: The label string. Defaults to "".
        confidence: A confidence value for the data. Defaults to 1.0.
        points: A list of lists of (x, y) points in [0, 1] x [0, 1].
    """
    
    def __init__(
        self,
        id_       : int   = -1,
        index     : int   = -1,
        label     : str   = "",
        confidence: float = 1.0,
        points    : list  = [],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"conf must be between 0.0 and 1.0, but got {confidence}."
            )
        if id_ <= 0 and label == "":
            raise ValueError(
                f"Either id or name must be defined, but got {id_} and {label}."
            )
        self.id_        = id_
        self.index      = index
        self.label      = label
        self.confidence = confidence
        self.points     = points

    @classmethod
    def from_value(cls, value: KeypointLabel | dict) -> KeypointLabel:
        """Create a :class:`KeypointLabel` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return KeypointLabel(**value)
        elif isinstance(value, KeypointLabel):
            return value
        else:
            raise ValueError(
                f"value must be a KeypointLabel class or a dict, but got "
                f"{type(value)}."
            )
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [
            self.points, self.id_, self.label, self.confidence, self.index
        ]


class KeypointsLabel(list[KeypointLabel], data.Label):
    """A list of keypoint labels for multiple objects in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        seq: A list of :class:`KeypointLabel` objects.
    """
    
    def __init__(self, seq: list[KeypointLabel | dict]):
        super().__init__(KeypointLabel.from_value(value=i) for i in seq)
    
    def __setitem__(self, index: int, item: KeypointLabel | dict):
        super().__setitem__(index, KeypointLabel.from_value(item))
    
    def insert(self, index: int, item: KeypointLabel | dict):
        super().insert(index, KeypointLabel.from_value(item))
    
    def append(self, item: KeypointLabel | dict):
        super().append(KeypointLabel.from_value(item))
    
    def extend(self, other: list[KeypointLabel | dict]):
        super().extend([KeypointLabel.from_value(item) for item in other])
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [i.data for i in self]
    
    @property
    def ids(self) -> list[int]:
        return [i.id_ for i in self]  
    
    @property
    def labels(self) -> list[str]:
        return [i.label for i in self]  

    @property
    def points(self) -> list:
        return [i.points for i in self]  
    

class COCOKeypointsLabel(KeypointsLabel):
    """A list of keypoint labels for multiple objects in COCO format.
    
    See Also: :class:`KeypointsLabel`.
    """
    pass

# endregion


# region Polyline

class PolylineLabel(data.Label):
    """A set of semantically related polylines or polygons for a single object
    in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        id_: The class ID of the polyline data. Defaults to -1 means unknown.
        index: An index for the polyline. Defaults to -1.
        label: The label string. Defaults to "".
        confidence: A confidence value for the data. Defaults to 1.0.
        points: A list of lists of (x, y) points in `[0, 1] x [0, 1]` describing
            the vertices of each shape in the polyline.
        closed: Whether the shapes are closed, in other words, and edge should
            be drawn. from the last vertex to the first vertex of each shape.
            Defaults to False.
        filled: Whether the polyline represents polygons, i.e., shapes that
            should be filled when rendering them. Defaults to False.
    """
    
    def __init__(
        self,
        id_       : int   = -1,
        index     : int   = -1,
        label     : str   = "",
        confidence: float = 1.0,
        points    : list  = [],
        closed    : bool  = False,
        filled    : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"conf must be between 0.0 and 1.0, but got {confidence}."
            )
        if id_ <= 0 and label == "":
            raise ValueError(
                f"Either id or name must be defined, but got {id_} and {label}."
            )
        self.id_        = id_
        self.index      = index
        self.label      = label
        self.closed     = closed
        self.filled     = filled
        self.confidence = confidence
        self.points     = points
       
    @classmethod
    def from_mask(
        cls,
        mask     : np.ndarray,
        label    : str = "",
        tolerance: int = 2,
        **kwargs
    ) -> PolylineLabel:
        """Create a :class:`PolylineLabel` instance with its :param:`mask`
        attribute populated from the given full image mask. The instance mask
        for the object is extracted by computing the bounding rectangle of the
        non-zero values in the image mask.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            label: A label string. Defaults to ““.
            tolerance: A tolerance, in pixels, when generating approximate
                polygons for each region. Typical values are 1-3 pixels.
                Defaults to 2.
            **kwargs: additional attributes for the :class:`PolylineLabel`.
        
        Return:
            A :class:`PolylineLabel` object.
        """
        pass
    
    @classmethod
    def from_value(cls, value: PolylineLabel | dict) -> PolylineLabel:
        """Create a :class:`PolylineLabel` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return PolylineLabel(**value)
        elif isinstance(value, PolylineLabel):
            return value
        else:
            raise ValueError(
                f"value must be a PolylineLabel class or a dict, but got "
                f"{type(value)}."
            )
        
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [
            self.points, self.id_, self.label, self.confidence, self.index
        ]
    
    def to_detection(
        self,
        mask_size : int | list[int] | None = None,
        image_size: int | list[int] | None = None,
    ) -> DetectionLabel:
        """Return a :class:`DetectionLabel` object of this instance whose
        bounding bbox tightly encloses the polyline. If a :param:`mask_size` is
        provided, an instance mask of the specified size encoding the polyline
        shape is included.
       
        Alternatively, if a :param:`frame_size` is provided, the required mask
        size is then computed based off the polyline points and
        :param:`frame_size`.
        
        Args:
            mask_size: An optional shape at which to render an instance mask
                for the polyline.
            image_size: Used when no :param:`mask_size` is provided. An optional
                shape of the frame containing this polyline that's used to
                compute the required :param:`mask_size`.
        
        Return:
            A :class:`DetectionLabel` object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : np.ndarray      | None = None,
        image_size: int | list[int] | None = None,
        target    : int                    = 255,
        thickness : int                    = 1,
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this class. Only
        object with instance masks (i.e., their :param:`mask` attributes
        populated) will be rendered.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            image_size: The shape of the segmentation mask to render. This
                parameter has no effect if a :param:`mask` is provided. Defaults
                to None.
            target: The pixel value to use to render the object. If you want
                color mask, just pass in the :param:`id` attribute. Defaults to
                255.
            thickness: The thickness, in pixels, at which to render (non-filled)
                polylines. Defaults to 1.
                
        Return:
            A :class:`SegmentationLabel` object.
        """
        pass


class PolylinesLabel(list[PolylineLabel], data.Label):
    """A list of polylines or polygon labels for multiple objects in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        seq: A list of :class:`PolylineLabel` objects.
    """

    def __init__(self, seq: list[PolylineLabel | dict]):
        super().__init__(PolylineLabel.from_value(value=i) for i in seq)

    def __setitem__(self, index: int, item: PolylineLabel | dict):
        super().__setitem__(index, PolylineLabel.from_value(item))

    def insert(self, index: int, item: PolylineLabel | dict):
        super().insert(index, PolylineLabel.from_value(item))

    def append(self, item: PolylineLabel | dict):
        super().append(PolylineLabel.from_value(item))

    def extend(self, other: list[PolylineLabel | dict]):
        super().extend([PolylineLabel.from_value(item) for item in other])
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [i.data for i in self]
    
    @property
    def ids(self) -> list[int]:
        return [i.id_ for i in self]
    
    @property
    def labels(self) -> list[str]:
        return [i.label for i in self]

    @property
    def points(self) -> list:
        return [i.points for i in self]
    
    def to_detections(
        self,
        mask_size : int | list[int] | None = None,
        image_size: int | list[int] | None = None,
    ) -> DetectionsLabel:
        """Return a :class:`DetectionsLabel` object of this instance whose
        bounding boxes tightly enclose the polylines. If a :param:`mask_size`
        is provided, an instance mask of the specified size encoding the
        polyline shape is included.
       
        Alternatively, if a :param:`frame_size` is provided, the required mask
        size is then computed based off the polyline points and
        :param:`frame_size`.
        
        Args:
            mask_size: An optional shape at which to render an instance mask
                for the polyline.
            image_size: Used when no :param:`mask_size` is provided. An optional
                shape of the frame containing this polyline that is used to
                compute the required :param:`mask_size`.
        
        Return:
            A :class:`DetectionsLabel` object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : np.ndarray      | None = None,
        image_size: int | list[int] | None = None,
        target    : int                    = 255,
        thickness : int                    = 1,
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this instance. Only
        polylines with instance masks (i.e., their :param:`mask` attributes
        populated) will be rendered.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            image_size: The shape of the segmentation mask to render. This
                parameter has no effect if a :param:`mask` is provided. Defaults
                to None.
            target: The pixel value to use to render the object. If you want
                color mask, just pass in the :param:`id` attribute. Defaults to
                255.
            thickness: The thickness, in pixels, at which to render (non-filled)
                polylines. Defaults to 1.
                
        Return:
            A :class:`SegmentationLabel` object.
        """
        pass

# endregion


# region Regression

class RegressionLabel(data.Label):
    """A single regression value.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        value: The regression value.
        confidence: A confidence value for the data. Defaults to 1.0.
    """
    
    def __init__(
        self,
        value     : float,
        confidence: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"conf must be between 0.0 and 1.0, but got {confidence}."
            )
        self.value      = value
        self.confidence = confidence
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [self.value]
    
# endregion


# region Segmentation

class SegmentationLabel(data.Label):
    """A semantic segmentation label in an image.
    
    See Also: :class:`mon.coreml.data.label.Label`.
    
    Args:
        id_: The ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: The name of the image. Defaults to None.
        path: The path to the image file. Defaults to None.
        mask: The image with integer values encoding the semantic labels.
            Defaults to None.
        load_on_create: If True, the image will be loaded into memory when the
            object is created. Defaults to False.
        keep_in_memory: If True, the image will be loaded into memory and kept
            there. Defaults to False.
    """

    to_rgb   : bool = True
    to_tensor: bool = False
    normalize: bool = False
    
    def __init__(
        self,
        id_           : int                 = uuid.uuid4().int,
        name          : str          | None = None,
        path          : pathlib.Path | None = None,
        mask          : np.ndarray   | None = None,
        load_on_create: bool                = False,
        keep_in_memory: bool                = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.id_            = id_
        self.image          = None
        self.keep_in_memory = keep_in_memory
        
        if path is not None:
            path = pathlib.Path(path)
            if not path.is_image_file():
                raise ValueError(
                    f"path must be a valid path to an image file, but got {path}."
                )
        self.path = path
        
        if name is None:
            name = str(pathlib.Path(path).name) if path.is_image_file() else f"{id_}"
        self.name = name

        if load_on_create and mask is None:
            mask = self.load()
        
        self.shape = mi.get_image_shape(image=mask) if mask is not None else None
       
        if self.keep_in_memory:
            self.mask = mask
    
    def load(
        self,
        path          : pathlib.Path | None = None,
        keep_in_memory: bool                = False,
    ) -> np.ndarray:
        """Load segmentation mask image into memory.
        
        Args:
            path: The path to the segmentation mask file. Defaults to None.
            keep_in_memory: If True, the image will be loaded into memory and
                kept there. Defaults to False.
            
        Return:
            Return image of shape HWC.
        """
        self.keep_in_memory = keep_in_memory
        
        if path is not None:
            path = pathlib.Path(path)
            if path.is_image_file():
                self.path = path
        if not path.is_image_file():
            raise ValueError(
                f"path must be a valid path to an image file, but got {path}."
            )
        
        mask = mi.read_image(
            path      = self.path,
            to_rgb    = self.to_rgb,
            to_tensor = self.to_tensor,
            normalize = self.normalize,
        )
        self.shape = mi.get_image_shape(image=mask) if (mask is not None) else self.shape
        
        if self.keep_in_memory:
            self.mask = mask
        return mask
        
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object."""
        return {
            "id"   : self.id_,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }
    
    @property
    def data(self) -> np.ndarray | None:
        """The label's data."""
        if self.mask is None:
            return self.load()
        else:
            return self.mask

# endregion
