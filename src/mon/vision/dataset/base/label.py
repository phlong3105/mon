#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple label types used in vision tasks and
datasets. We try to support all possible data types: :class:`torch.Tensor`,
:class:`np.ndarray`, or :class:`Sequence`, but we prioritize
:class:`torch.Tensor`.
"""

from __future__ import annotations

__all__ = [
    "ClassLabel", "ClassLabels", "COCODetectionsLabel", "COCOKeypointsLabel",
    "ClassificationLabel", "ClassificationsLabel", "DetectionLabel",
    "DetectionsLabel", "HeatmapLabel", "ImageLabel", "KITTIDetectionsLabel",
    "KeypointLabel", "KeypointsLabel", "PolylineLabel", "PolylinesLabel",
    "RegressionLabel", "SegmentationLabel", "TemporalDetectionLabel",
    "VOCDetectionsLabel", "YOLODetectionsLabel",
]

import uuid
from typing import Sequence

import numpy as np
import torch

from mon import core, coreimage as ci, coreml
from mon.vision import constant
from mon.vision.typing import (
    BBoxType, DictType, Image, IntAnyT, LogitsType, MaskType, PathType,
    PointsType, VisionBackendType,
)

ClassLabel  = coreml.ClassLabel
ClassLabels = coreml.ClassLabels


# region Classification

class ClassificationLabel(coreml.Label):
    """A classification label for an image.
    
    Args:
        id: A class ID of the classification label. Defaults to -1 means
            unknown.
        label: A label string. Defaults to ““.
        confidence: A confidence value in [0.0, 1.0] for the classification.
            Defaults to 1.0.
        logits: Logits associated with the labels. Defaults to None.
    """
    
    def __init__(
        self,
        id        : int               = -1,
        label     : str               = "",
        confidence: float             = 1.0,
        logits    : LogitsType | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 <= confidence <= 1.0
        assert id >= 0 or label != "", \
            f"Either :param:`id` or :param:`name` must be defined. " \
            f"But got: {id} and {label}."
        self.id         = id
        self.label      = label
        self.logits     = logits
        self.confidence = confidence
        
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.tensor([self.id], dtype=torch.int64)
        

class ClassificationsLabel(coreml.Label):
    """A list of classification labels for an image. It is used for multi-labels
    or multi-classes classification tasks.

    Args:
        classifications: A list of :class:`ClassificationLabel` objects.
        logits: Logits associated with the labels.
    """
    
    def __init__(
        self,
        classifications: Sequence[ClassificationLabel],
        logits         : LogitsType,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(classifications, list | tuple) \
               and all(isinstance(c, ClassificationLabel) for c in classifications)
        self.classifications = classifications
        self.logits          = logits
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.stack([c.tensor for c in self.classifications], dim=0)

# endregion


# region Object Detection

class DetectionLabel(coreml.Label):
    """An object detection label. Usually, it is represented as a list of
    bounding boxes (for an object with multiple parts created by an occlusion),
    and an instance mask.
    
    Args:
        index: An index for the object. Defaults to -1.
        id: A class ID of the detection label. Defaults to -1 means unknown.
        label: Label string. Defaults to "".
        bbox: A list of relative bounding boxes' coordinates in the range of
            [0.0, 1.0] in the normalized xywh format.
        mask: Instance segmentation masks for the object within its bounding
            box, which should be a binary (0/1) 2D sequence or a binary integer
            tensor. Defaults to None
        confidence: A confidence value in [0.0, 1.0] for the detection. Defaults
            to 1.0.
    """
    
    def __init__(
        self,
        index     : int             = -1,
        id        : int             = -1,
        label     : str             = "",
        bbox      : BBoxType        = [],
        mask      : MaskType | None = None,
        confidence: float           = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 <= confidence <= 1.0
        self.index      = index
        self.id         = id
        self.label      = label
        self.mask       = mask
        self.confidence = confidence
        
        if isinstance(bbox, np.ndarray):
            bbox = torch.from_numpy(bbox)
        elif isinstance(bbox, list | tuple):
            bbox = torch.FloatTensor(bbox)
        self.bbox = bbox

    @classmethod
    def from_mask(cls, mask: MaskType, label: str, **kwargs) -> DetectionLabel:
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
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.FloatTensor(
            [
                self.index, self.id, self.bbox[0], self.bbox[1], self.bbox[2],
                self.bbox[3], self.confidence,
            ]
        )
        pass
    
    def to_polyline(
        self,
        tolerance: int  = 2,
        filled   : bool = True
    ) -> PolylineLabel:
        """Return a :class:`PolylineLabel` object of this instance. If the
        detection has a mask, the returned polyline will trace the boundary of
        the mask. Otherwise, the polyline will trace the bounding box itself.
        
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
        mask      : MaskType | None = None,
        frame_size: IntAnyT  | None = None,
        target    : int             = 255
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this instance. The
        detection must have an instance mask, i.e., :param:`mask` attribute must
        be populated. You must give either :param:`mask` or :param:`frame_size`
        to use this method.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            frame_size: The shape of the segmentation mask to render. This
                parameter has no effect if a :param:`mask` is provided. Defaults
                to None.
            target: The pixel value to use to render the object. If you want
                color mask, just pass in the :param:`id` attribute. Defaults to
                255.
        
        Return:
            A :class:`SegmentationLabel` object.
        """
        raise NotImplementedError(f"This function has not been implemented!")


class DetectionsLabel(coreml.Label):
    """A list of object detection labels in an image.
    
    Args:
        detections: A list of :class:`DetectionLabel` objects.
    """
    
    def __init__(self, detections: Sequence[DetectionLabel], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(detections, list | tuple) \
               and all(isinstance(c, DetectionLabel) for c in detections)
        self.detections = detections
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.stack([d.tensor for d in self.detections], dim=0)
    
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
        print(f"This function has not been implemented!")
    
    def to_segmentation(
        self,
        mask      : MaskType | None = None,
        frame_size: IntAnyT  | None = None,
        target    : int             = 255
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this instance. Only
        detections with instance masks (i.e., their :param:`mask` attributes
        populated) will be rendered.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            frame_size: The shape of the segmentation mask to render. This
                parameter has no effect if a :param:`mask` is provided. Defaults
                to None.
            target: The pixel value to use to render the object. If you want
                color mask, just pass in the :param:`id` attribute. Defaults to
                255.
        
        Return:
            A :class:`SegmentationLabel` object.
        """
        print(f"This function has not been implemented!")


class COCODetectionsLabel(DetectionsLabel):
    """A list of object detection labels in COCO format."""
    pass


class KITTIDetectionsLabel(DetectionsLabel):
    """A list of object detection labels in KITTI format."""
    pass


class VOCDetectionsLabel(DetectionsLabel):
    """A list of object detection labels in VOC format. One VOCDetections
    corresponds to one image and one annotation `.xml` file.
    
    Args:
        folder: The folder that contains the images.
        filename: Name of the physical file that exists in the folder.
        path: Absolute path where the image file is present.
        source: Specifies the original location of the file in a database. Since
            we don't use a database, it is set to “Unknown” by default.
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
            - truncated: Indicates that the bounding box specified for the
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
        folder     : str                = "",
        filename   : str                = "",
        path       : PathType           = "",
        source     : DictType           = {"database": "Unknown"},
        size       : DictType           = {"width": 0, "height": 0, "depth": 3},
        segmented  : int                = 0,
        classlabels: ClassLabels | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.folder      = folder
        self.filename    = filename
        self.path        = core.Path(path)
        self.source      = source
        self.size        = size
        self.segmented   = segmented
        self.classlabels = classlabels
    
    @classmethod
    def from_file(
        cls,
        path       : PathType,
        classlabels: ClassLabels | None = None
    ) -> VOCDetectionsLabel:
        """Create a :class:`VOCDetections` object from a `.xml` file.
        
        Args:
            path: Path to the `.xml` file.
            classlabels: :class:`ClassLabels` object. Defaults to None.
            
        Return:
            A :class:`VOCDetections` object.
        """
        path = core.Path(path)
        assert path.is_xml_file()
        
        xml_data = core.load_from_file(path)
        assert "annotation" in xml_data
       
        annotation = xml_data["annotation"]
        folder     = annotation.get("folder",   "")
        filename   = annotation.get("file_name", "")
        image_path = annotation.get("path",     "")
        source     = annotation.get("source",   {"database": "Unknown"})
        size       = annotation.get("size",     {"width": 0, "height": 0, "depth": 3})
        width      = int(size.get("width" , 0))
        height     = int(size.get("height", 0))
        depth      = int(size.get("depth" , 0))
        segmented  = annotation.get("segmented", 0)
        objects    = annotation.get("object"   , [])
        objects    = [objects] if not isinstance(objects, list) else objects
        
        detections: list[DetectionLabel] = []
        for i, o in enumerate(objects):
            name       = o.get["name"]
            bndbox     = o.get["bndbox"]
            bbox       = torch.FloatTensor([bndbox["xmin"], bndbox["ymin"],
                                            bndbox["xmax"], bndbox["ymax"]])
            bbox       = ci.box_xyxy_to_cxcywhn(box=bbox, height=height, width=width)
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
                DetectionLabel(
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


class YOLODetectionsLabel(DetectionsLabel):
    """A list of object detection labels in YOLO format. YOLO label consists of
    several bounding boxes. One YOLO label corresponds to one image and one
    annotation file. """
    
    @classmethod
    def from_file(cls, path: PathType) -> YOLODetectionsLabel:
        """Create a :class:`YOLODetectionsLabel` object from a `.txt` file.
        
        Args:
            path: Path to the annotation `.txt` file.
        
        Return:
            A :class:`YOLODetections` object.
        """
        path = core.Path(path)
        assert path.is_txt_file()
        
        detections: list[DetectionLabel] = []
        lines = open(path, "r").readlines()
        for l in lines:
            d          = l.split(" ")
            bbox       = [float(b) for b in d[1:5]]
            confidence = float(d[5]) if len(d) >= 6 else 1.0
            detections.append(
                DetectionLabel(
                    id         = int(d[0]),
                    bbox       = bbox,
                    confidence = confidence
                )
            )
        return cls(detections=detections)
        

class TemporalDetectionLabel(coreml.Label):
    """An object detection label in a video whose support is defined by a start
    and end frame. Usually, it is represented as a list of bounding boxes (for
    an object with multiple parts created by an occlusion), and an instance
    mask.
    """
    
    @property
    def tensor(self) -> torch.Tensor:
        raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Heatmap

class HeatmapLabel(coreml.Label):
    """A heatmap label in an image.
    
    Args:
        map: A 2D numpy array.
        range: An optional [min, max] range of the map's values. If None is
            provided, [0, 1] will be assumed if :param:`map` contains floating
            point values, and [0, 255] will be assumed if :param:`map` contains
            integer values.
    """

    @property
    def tensor(self) -> torch.Tensor:
        raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Image

class ImageLabel(coreml.Label):
    """A ground-truth image label for an image.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
    
    Args:
        id: An ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: A name of the image. Defaults to None.
        path: A path to the image file. Defaults to None.
        image: A ground-truth image to be loaded. Defaults to None.
        load_on_create: If True, the image will be loaded into memory when the
            object is created. Defaults to False.
        keep_in_memory: If True, the image will be loaded into memory and kept
            there. Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
    """
    
    to_rgb   : bool = True
    to_tensor: bool = True
    normalize: bool = True
    
    def __init__(
        self,
        id            : int                 = uuid.uuid4().int,
        name          : str          | None = None,
        path          : PathType     | None = None,
        image         : Image        | None = None,
        load_on_create: bool                = False,
        keep_in_memory: bool                = False,
        backend       : VisionBackendType   = constant.VISION_BACKEND,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.id             = id
        self.image          = None
        self.keep_in_memory = keep_in_memory
        self.backend        = backend
        
        if path is not None:
            path = core.Path(path)
            assert path.is_image_file()
        self.path: core.Path = path
        
        if name is None:
            name = str(core.Path(path).name) \
                if path.is_image_file() else f"{id}"
        self.name = name
        
        if load_on_create and image is None:
            image = self.load()

        self.shape = ci.get_image_shape(image=image) \
            if image is not None else None

        if self.keep_in_memory:
            self.image = image
    
    def load(
        self,
        path          : PathType | None = None,
        keep_in_memory: bool            = False,
    ) -> torch.Tensor:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Defaults to None.
            keep_in_memory: If True, the image will be loaded into memory and
                kept there. Defaults to False.
            
        Return:
            An image Tensor of shape [1, C, H, W] to caller.
        """
        self.keep_in_memory = keep_in_memory
        
        if path is not None and core.is_image_file(path=path):
            self.path = core.Path(path)
        assert self.path.is_image_file()
        
        image = ci.read_image(
            path      = self.path,
            to_rgb    = self.to_rgb,
            to_tensor = self.to_tensor,
            normalize = self.normalize,
            backend   = self.backend
        )
        self.shape = ci.get_image_shape(image=image) \
            if (image is not None) else self.shape
        
        if self.keep_in_memory:
            self.image = image
        
        return image
        
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "id"   : self.id,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        if self.keep_in_memory:
            return self.image
        else:
            return self.load()

# endregion


# region Keypoint

class KeypointLabel(coreml.Label):
    """A list keypoints label for a single object in an image.
    
    Args:
        index: An index for the polyline. Defaults to -1.
        id: The class ID of the polyline label. Defaults to -1 means unknown.
        label: The label string. Defaults to "".
        points: A list of lists of (x, y) points in [0, 1] x [0, 1].
        confidence: A confidence in [0.0, 1.0] for the detection.
            Defaults to 1.0.
    """
    
    def __init__(
        self,
        index     : int        = -1,
        id        : int        = -1,
        label     : str        = "",
        points    : PointsType = [],
        confidence: float      = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 <= confidence <= 1.0
        self.index      = index
        self.id         = id
        self.label      = label
        self.confidence = confidence
        
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        elif isinstance(points, list | tuple):
            points = torch.FloatTensor(points)
        self.points = points
       
    @property
    def tensor(self) -> torch.Tensor:
        raise NotImplementedError(f"This function has not been implemented!")


class KeypointsLabel(coreml.Label):
    """A list of keypoint labels for multiple objects in an image.
    
    Args:
        keypoints: A list of :class:`KeypointLabel` objects.
    """
    
    def __init__( self, keypoints: Sequence[KeypointLabel], args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(keypoints, list | tuple) \
               and all(isinstance(c, KeypointLabel) for c in keypoints)
        self.keypoints = keypoints
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.stack([k.tensor for k in self.keypoints], dim=0)


class COCOKeypointsLabel(KeypointsLabel):
    """A list of keypoint labels for multiple objects in COCO format."""
    pass

# endregion


# region Polyline

class PolylineLabel(coreml.Label):
    """A set of semantically related polylines or polygons for a single object
    in an image.
    
    Args:
        index: An index for the polyline. Defaults to -1.
        id: The class ID of the polyline label. Defaults to -1 means unknown.
        label: The label string. Defaults to "".
        points: A list of lists of (x, y) points in `[0, 1] x [0, 1]` describing
            the vertices of each shape in the polyline.
        closed: Whether the shapes are closed, in other words, and edge should
            be drawn.
        from the last vertex to the first vertex of each shape. Defaults to
            False.
        filled: Whether the polyline represents polygons, i.e., shapes that
            should be filled when rendering them. Defaults to False.
        confidence: A confidence in [0.0, 1.0] for the detection. Defaults to
            1.0.
    """
    
    def __init__(
        self,
        index     : int        = -1,
        id        : int        = -1,
        label     : str        = "",
        points    : PointsType = [],
        closed    : bool       = False,
        filled    : bool       = False,
        confidence: float      = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 <= confidence <= 1.0
        self.index      = index
        self.id         = id
        self.label      = label
        self.closed     = closed
        self.filled     = filled
        self.confidence = confidence
        
        if not isinstance(points, torch.Tensor):
            points = torch.FloatTensor(points)
        self.points = points
       
    @classmethod
    def from_mask(
        cls,
        mask     : MaskType,
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
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        raise NotImplementedError(f"This function has not been implemented!")
    
    def to_detection(
        self,
        mask_size : IntAnyT | None = None,
        frame_size: IntAnyT | None = None,
    ) -> DetectionLabel:
        """Return a :class:`DetectionLabel` object of this instance whose
        bounding box tightly encloses the polyline. If a :param:`mask_size` is
        provided, an instance mask of the specified size encoding the polyline
        shape is included.
       
        Alternatively, if a :param:`frame_size` is provided, the required mask
        size is then computed based off the polyline points and
        :param:`frame_size`.
        
        Args:
            mask_size: An optional shape at which to render an instance mask
                for the polyline.
            frame_size: Used when no :param:`mask_size` is provided. An optional
                shape of the frame containing this polyline that's used to
                compute the required :param:`mask_size`.
        
        Return:
            A :class:`DetectionLabel` object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : MaskType | None = None,
        frame_size: IntAnyT  | None = None,
        target    : int             = 255,
        thickness : int             = 1,
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this class. Only
        object with instance masks (i.e., their :param:`mask` attributes
        populated) will be rendered.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            frame_size: The shape of the segmentation mask to render. This
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


class PolylinesLabel(coreml.Label):
    """A list of polylines or polygon labels for multiple objects in an image.
    
    Args:
        polylines: A list of :class:`PolylineLabel` objects.
    """
    
    def __init__(self, polylines: Sequence[PolylineLabel], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(polylines, list) \
               and all(isinstance(p, PolylineLabel) for p in polylines)
        self.polylines = polylines
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.stack([p.tensor for p in self.polylines], dim=0)
    
    def to_detections(
        self,
        mask_size : IntAnyT | None = None,
        frame_size: IntAnyT | None = None,
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
            frame_size: Used when no :param:`mask_size` is provided. An optional
                shape of the frame containing this polyline that is used to
                compute the required :param:`mask_size`.
        
        Return:
            A :class:`DetectionsLabel` object.
        """
        pass
    
    def to_segmentation(
        self,
        mask      : MaskType | None = None,
        frame_size: IntAnyT  | None = None,
        target    : int             = 255,
        thickness : int             = 1,
    ) -> SegmentationLabel:
        """Return a :class:`SegmentationLabel` object of this instance. Only
        polylines with instance masks (i.e., their :param:`mask` attributes
        populated) will be rendered.
        
        Args:
            mask: An optional 2D integer numpy array to use as an initial mask
                to which to add this object. Defaults to None.
            frame_size: The shape of the segmentation mask to render. This
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

class RegressionLabel(coreml.Label):
    """A single regression value.
    
    Args:
        value: The regression value.
        confidence: A confidence in [0.0, 1.0] for the classification. Defaults
            to 1.0.
    """
    
    def __init__(
        self,
        value     : float,
        confidence: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 0.0 <= confidence <= 1.0
        self.value      = value
        self.confidence = confidence
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        return torch.FloatTensor([self.value])
    
# endregion


# region Segmentation

class SegmentationLabel(coreml.Label):
    """A semantic segmentation label in an image.

    Args:
        id: The ID of the image. This can be an integer or a string. This
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
        backend: The image processing backend. Defaults to VISION_BACKEND.
    """

    to_rgb   : bool = True
    to_tensor: bool = True
    normalize: bool = True
    
    def __init__(
        self,
        id            : int               = uuid.uuid4().int,
        name          : str      | None   = None,
        path          : PathType | None   = None,
        mask          : MaskType | None   = None,
        load_on_create: bool              = False,
        keep_in_memory: bool              = False,
        backend       : VisionBackendType = constant.VISION_BACKEND,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.id             = id
        self.image          = None
        self.keep_in_memory = keep_in_memory
        self.backend        = backend
        
        if path is not None:
            path = core.Path(path)
            assert path.is_image_file()
        self.path: core.Path = path
        
        if name is None:
            name = str(core.Path(path).name) \
                if path.is_image_file() else f"{id}"
        self.name = name
        
        if load_on_create and mask is None:
            mask = self.load()

        self.shape = ci.get_image_shape(image=mask) \
            if mask is not None else None

        if self.keep_in_memory:
            self.mask = mask
    
    def load(
        self,
        path          : PathType | None = None,
        keep_in_memory: bool            = False,
    ) -> torch.Tensor:
        """Load segmentation mask image into memory.
        
        Args:
            path: The path to the segmentation mask file. Defaults to None.
            keep_in_memory: If True, the image will be loaded into memory and
                kept there. Defaults to False.
            
        Return:
            Return image Tensor of shape [1, C, H, W] to caller.
        """
        self.keep_in_memory = keep_in_memory
        
        if path.is_image_file():
            self.path = core.Path(path)
        assert self.path.is_image_file()
        
        mask = ci.read_image(
            path      = self.path,
            to_rgb    = self.to_rgb,
            to_tensor = self.to_tensor,
            normalize = self.normalize,
            backend   = self.backend
        )
        self.shape = ci.get_image_shape(image=mask) \
            if (mask is not None) else self.shape
        
        if self.keep_in_memory:
            self.mask = mask
        
        return mask
        
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object."""
        return {
            "id"   : self.id,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }
    
    @property
    def tensor(self) -> torch.Tensor:
        """The label in :class:`torch.Tensor` format."""
        if self.mask is None:
            self.load()
        return self.mask

# endregion
