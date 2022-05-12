#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label file handler for loading and dumping labels from/to our custom label
format.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from typing import Optional

import numpy as np

from one.core import LABEL_HANDLERS
from one.data.data_class import ImageInfo
from one.data.data_class import ObjectAnnotation
from one.data.data_class import VisionData
from one.data.label_handler.base import BaseLabelHandler
from one.imgproc import box_cxcywh_norm_to_xyxy
from one.imgproc import compute_box_area
from one.io import dump
from one.io import is_json_file
from one.io import load

__all__ = [
    "VisionDataHandler"
]


# MARK: - VisionDataHandler

@LABEL_HANDLERS.register(name="custom")
@LABEL_HANDLERS.register(name="default")
@LABEL_HANDLERS.register(name="visual_data")
class VisionDataHandler(BaseLabelHandler):
    """Handler for loading and dumping labels from/to our custom label
    format defined in `onedataset.core.vision_data`.
    """

    # MARK: Load

    def load_from_file(
        self,
        image_path   : str,
        label_path   : str,
        semantic_path: Optional[str] = None,
        eimage_path  : Optional[str] = None,
        **kwargs
    ) -> VisionData:
        """Load data from file.

        Args:
            image_path (str):
                Image filepath.
            label_path (str):
                Label filepath.
            semantic_path (str, optional):
                Semantic segmentation image filepath.
            eimage_path (str, optional):
                Enhanced image filepath.

        Return:
            visual_data (VisionData):
                A `VisualData` item.
        """
        # NOTE: Load content from file
        l = load(label_path) if is_json_file(label_path) else None

        image_info = ImageInfo()
        if hasattr(l, "image_info"):
            image_info = ImageInfo(*l["image_info"])

        semantic_info = None
        if hasattr(l, "semantic_info"):
            semantic_info = ImageInfo(*l["semantic_info"])

        eimage_info = None
        if hasattr(l, "eimage_info"):
            eimage_info = ImageInfo(*l["eimage_info"])

        shape0 = image_info.shape0

        # NOTE: Parse image info
        image_info = ImageInfo.from_file(image_path, image_info)
        if semantic_path and semantic_info:
            semantic_info = ImageInfo.from_file(semantic_path, semantic_info)
        if eimage_info and eimage_info:
            eimage_info = ImageInfo.from_file(eimage_path, eimage_info)

        if l is None or not isinstance(l, dict):
            return VisionData(
                image_info    = image_info,
                semantic_info = semantic_info,
                eimage_info   = eimage_info
            )

        # NOTE: Parse all annotations
        objs = []
        for o in l["objects"]:
            obj      = ObjectAnnotation(*o)
            obj.box  = np.array(obj.box, np.float32)
            box_xyxy = box_cxcywh_norm_to_xyxy(obj.box, shape0[0], shape0[1])
            if obj.area == 0.0:
                obj.area = compute_box_area(box_xyxy)
            objs.append(obj)

        return VisionData(
            image_info    = image_info,
            semantic_info = semantic_info,
            eimage_info   = eimage_info,
            objects       = objs
        )

    # MARK: Dump

    def dump_to_file(self, data: VisionData, path: str, **kwargs):
        """Dump data from object to file.

        Args:
            data (VisionData):
                Visual data item.
            path (str):
                Label filepath to dump the data.
        """
        # NOTE: Prepare output data
        data_out                         = asdict(data)
        ordered_data                     = OrderedDict()
        ordered_data["image_info"]       = data_out["image_info"]
        ordered_data["image_annotation"] = data_out["image_annotation"]
        ordered_data["semantic_info"]    = data_out["semantic_info"]
        ordered_data["instance_info"]    = data_out["instance_info"]
        ordered_data["panoptic_info"]    = data_out["panoptic_info"]
        ordered_data["eimage_info"]      = data_out["eimage_info"]
        objects                          = data_out["objects"].copy()
        for i, obj in enumerate(objects):
            objects[i]["box"] = obj["box"].tolist()
        ordered_data["objects"] = objects

        # NOTE: Dump to file
        dump(obj=data_out, path=path, file_format="json", indent=4)
