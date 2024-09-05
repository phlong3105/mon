#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mask.

This module implements utility functions for different types of masks.
"""

from __future__ import annotations

__all__ = [
    "depth_map_to_color",
	"label_map_id_to_train_id",
    "label_map_color_to_id",
    "label_map_id_to_color",
]

import cv2
import numpy as np

from mon.core.image import utils


# region Conversion

def depth_map_to_color(
    depth_map: np.ndarray,
    color_map: int = cv2.COLORMAP_JET,
    use_rgb  : bool = False,
) -> np.ndarray:
    """Convert depth map to color-coded images.
    
    Args:
        depth_map: A depth map of type :obj:`numpy.ndarray` in ``[H, W, 1]``
            format.
        color_map: A color map for the depth map. Default: ``cv2.COLORMAP_JET``.
        use_rgb: If ``True``, convert the heatmap to RGB format.
            Default: ``False``.
    """
    if utils.is_normalized_image(depth_map):
        depth_map = np.uint8(255 * depth_map)
    depth_map = cv2.applyColorMap(np.uint8(255 * depth_map), color_map)
    if use_rgb:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    return depth_map
    

def label_map_id_to_train_id(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from IDs to training IDs.
    
    Args:
        label_map: An label map of type :obj:`numpy.ndarray` in ``[H, W, C]``
            format.
        classlabels: A list of class-labels.
    """
    id2train_id = classlabels.id2train_id
    h, w        = utils.get_image_size(label_map)
    label_ids   = np.zeros((h, w), dtype=np.uint8)
    label_map   = label_map[:, :, 0] if label_map.ndim == 3 else label_map
    for id, train_id in id2train_id.items():
        label_ids[label_map == id] = train_id
    label_ids   = np.expand_dims(label_ids, axis=-1)
    return label_ids
 

def label_map_id_to_color(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map to color-coded images.
    
    Args:
        label_map: An label map of type :obj:`numpy.ndarray` in ``[H, W, C]``
            format.
        classlabels: A list of class-labels, each has predefined color.
    """
    id2color   = classlabels.id2color
    h, w       = utils.get_image_size(label_map)
    color_map  = np.zeros((h, w, 3), dtype=np.uint8)
    labels_ids = label_map[:, :, 0] if label_map.ndim == 3 else label_map
    for id, color in id2color.items():
        color_map[labels_ids == id] = color
    return color_map


def label_map_color_to_id(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert color-coded images to label map.

    Args:
        label_map: An color-coded images of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels, each has predefined color.
    """
    id2color  = classlabels.id2color
    h, w      = utils.get_image_size(label_map)
    label_ids = np.zeros((h, w), dtype=np.uint8)
    for id, color in id2color.items():
        label_ids[np.all(label_map == color, axis=-1)] = id
    label_ids = np.expand_dims(label_ids, axis=-1)
    return label_ids

# endregion
