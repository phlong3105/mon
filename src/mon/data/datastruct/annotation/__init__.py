#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements multiple annotation types. We try to support all
possible data types: :class:`torch.Tensor`, :class:`numpy.ndarray`, or
:class:`Sequence`, but we prioritize :class:`torch.Tensor`.

The term "annotation" is commonly used in machine learning and deep learning to
describe both ground truth label and model prediction. Basically, both of them
share similar data structure.
"""

from __future__ import annotations

from typing import Optional

import mon.data.datastruct.annotation.base
import mon.data.datastruct.annotation.bbox
import mon.data.datastruct.annotation.category
import mon.data.datastruct.annotation.classlabel
import mon.data.datastruct.annotation.image
import mon.data.datastruct.annotation.value
from mon import core
from mon.data.datastruct.annotation.base import *
from mon.data.datastruct.annotation.bbox import *
from mon.data.datastruct.annotation.category import *
from mon.data.datastruct.annotation.classlabel import *
from mon.data.datastruct.annotation.image import *
from mon.data.datastruct.annotation.value import *

console       = core.console
error_console = core.error_console


# region Utils

def get_albumentation_target_type(annotation) -> str | None:
    """Returns the type of target that Albumentations expects.
    One of: [``'image'``, ``'mask'``, ``'bboxes'``, ``'keypoints'``, ``'values'``].
    """
    if annotation in [ImageAnnotation]:
        return "image"
    elif annotation in [BBoxAnnotation, BBoxesAnnotation]:
        return "bboxes"
    elif annotation in [ClassificationAnnotation, RegressionAnnotation]:
        return "values"
    elif annotation in [SegmentationAnnotation]:
        return "mask"
    else:
        error_console.log(f"Unknown annotation type: {annotation}, {type(annotation)}")
        return None


class DatapointAttributes(dict[str: Optional[Annotation]]):
    """A dictionary of datapoint attributes with the keys are the attribute names
    and the values are the annotation types.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def to_tensor_fns(self) -> dict[str: Optional[callable]]:
        """Returns a dictionary of functions to convert the annotation to a tensor."""
        return {k: getattr(v, "to_tensor", None) for k, v in self.items() if v}
    
    def collate_fns(self) -> dict[str: Optional[callable]]:
        """Returns a dictionary of functions to collate the annotation."""
        return {k: getattr(v, "collate_fn", None) for k, v in self.items() if v}
    
    def albumentation_target_types(self) -> dict[str: str]:
        """Returns a dictionary of target types that Albumentations expects."""
        target_types = {k: get_albumentation_target_type(v) for k, v in self.items() if v}
        target_types = {k: v for k, v in target_types.items() if v}
        return target_types
    
    def get_tensor_fn(self, key: str) -> Optional[callable]:
        """Returns the function to convert the annotation to a tensor."""
        return self.to_tensor_fns().get(key, None)
    
    def get_collate_fn(self, key: str) -> Optional[callable]:
        """Returns the function to collate the annotation."""
        return self.collate_fns().get(key, None)
    
    def get_albumentation_target_type(self, key: str) -> Optional[str]:
        """Returns the target type that Albumentations expects."""
        return self.albumentation_target_types().get(key, None)
    
# endregion
