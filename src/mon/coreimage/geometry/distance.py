#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements distance functions. We try to support both
:class:`np.ndarray` and :class:`torch.Tensor`.

See Also:
    https://github.com/scipy/scipy/blob/v1.10.0/scipy/spatial/distance.py
"""

from __future__ import annotations

__all__ = [

]

from typing import Sequence

import numpy as np
import scipy
import torch

import mon.core
from mon.core import math
from mon.coreimage.typing import Number


# region Numeric Distance

def minkowski(
    u: np.ndarray | torch.Tensor | Sequence[Number],
    v: np.ndarray | torch.Tensor | Sequence[Number],
    o: int = 2,
    w: np.ndarray | torch.Tensor | Sequence[Number] | None = None,
) -> float | torch.Tensor | None:
    """Compute the Minkowski distance between two 1-D arrays.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        o: The order of the norm of the difference :math:`{\\|u-v\\|}_p`.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Minkowski distance between vectors :param:`u` and :param:`v`.
    """
    if isinstance(u, torch.Tensor) and type(u) == type(v):
        mon.core.error_console(f"This function has not been implemented.")
        return None
    elif type(u) == type(v):  # For other cases, we use :mod:`scipy` package.
        return scipy.spatial.distance.minkowski(u=u, v=v, p=o, w=w)
    else:
        raise TypeError(f":param:`u` and :param:`v` must be the same type.")


# endregion


def angle_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the angle of two vectors."""
    vec1 = np.array([x[-1][0] - x[0][0], x[-1][1] - x[0][1]])
    vec2 = np.array([y[-1][0] - y[0][0], y[-1][1] - y[0][1]])
    l1   = np.sqrt(vec1.dot(vec1))
    l2   = np.sqrt(vec2.dot(vec2))
    if l1 == 0 or l2 == 0:
        return False
    cos   = vec1.dot(vec2) / (l1 * l2)
    angle = np.arccos(cos) * 360 / (2 * np.pi)
    return angle
    

def chebyshev(x: np.ndarray, y: np.ndarray) -> float:
    """Chebyshev distance: a metric defined on a vector space where the distance
    between two vectors is the greatest of their differences along any
    coordinate dimension.
    """
    n   = x.shape[0]
    ret = -1 * np.inf
    for i in range(n):
        d = abs(x[i] - y[i])
        if d > ret:
            ret = d
    return ret


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the pair-wise cosine similarity between points.
    
    References:
        https://www.codestudyblog.com/cnb2001/0119184904.html
    
    Args:
        x: An matrix of N samples of dimensionality M.
        y: An matrix of L samples of dimensionality M.
        
    Returns:
        A matrix of size len(x), len(y) such that element (i, j) contains the
        squared distance between `x[i]` and `y[j]`.
    """
    """
    n = x.shape[0]
    xy_dot = 0.0
    x_norm = 0.0
    y_norm = 0.0
    for i in range(n):
        xy_dot += x[i] * y[i]
        x_norm += x[i] * x[i]
        y_norm += y[i] * y[i]
    return 1.0 - xy_dot / (sqrt(x_norm) * sqrt(y_norm))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise RuntimeError(
            f":param:`x` and :param:`y` shape must be matched. "
            f"But got: {x.shape} != {y.shape}."
        )
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    elif x.ndim != 2:
        raise RuntimeError(f":param:`x.ndim` must == 2. But got: {x.ndim}.")
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    return np.dot(x, y.T) / (x_norm * y_norm)


def cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pair-wise cosine distance between points in :param:`x` and
    :param:`y`.
    
    Args:
        x: An matrix of N samples of dimensionality M.
        y: An matrix of L samples of dimensionality M
    
    Returns:
        A matrix of size len(x), len(y) such that element (i, j) contains the
        squared distance between `x[i]` and `y[j]`.
    """
    return 1.0 - cosine_similarity(x, y)


def euclidean(
    x        : np.ndarray,
    y        : np.ndarray,
    keep_dims: bool = True
) -> float:
    """Compute pair-wise euclidean distance between points in `x` and `y`.
    
    Args:
        x (np.ndarray[N, M]):
            An matrix of N samples of dimensionality M.
        y (np.ndarray[L, M]):
            An matrix of L samples of dimensionality M.
        keep_dims (bool):
            If this is set to `True`, the axes which are normed over are left in
            the result as dimensions with size one. With this option the result
            will broadcast correctly against the original `x`.
        
    Returns:
        dist (np.ndarray):
            Returns a matrix of size len(x), len(y) such that element (i, j)
            contains the squared distance between `x[i]` and `y[j]`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise RuntimeError(f"`x` and `y` shape must be matched. "
                           f"But got: {x.shape} != {y.shape}.")
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    elif x.ndim != 2:
        raise RuntimeError(f"`x.ndim` must == 2. But got {x.ndim}.")
    
    return np.linalg.norm(x - y, ord=2, keepdims=keep_dims)


def hausdorff(x: np.ndarray, y: np.ndarray) -> float:
    """Calculation of Hausdorff distance btw 2 arrays.
    
    `euclidean_distance`, `manhattan_distance`, `chebyshev_distance`,
    `cosine_distance`, `haversine_distance` could be use for this function.
    """
    cmax = 0.0
    for i in range(len(x)):
        cmin = np.inf
        for j in range(len(y)):
            d = euclidean_distance(x[i, :], y[j, :])
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmax < cmin < np.inf:
            cmax = cmin
    return cmax


def haversine(x: np.ndarray, y: np.ndarray) -> float:
    """Haversine (or great circle) distance is the angular distance between two
    points on the surface of a sphere. First coordinate of each point is assumed
    to be the latitude, the second is the longitude, given in radians.
    Dimension of the data must be 2.
    """
    R 		= 6378.0
    radians = np.pi / 180.0
    lat_x 	= radians * x[0]
    lon_x 	= radians * x[1]
    lat_y 	= radians * y[0]
    lon_y 	= radians * y[1]
    dlon  	= lon_y - lon_x
    dlat  	= lat_y - lat_x
    a 		= (pow(math.sin(dlat / 2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon / 2.0), 2.0))
    return R * 2 * math.asin(math.sqrt(a))


def manhattan(x: np.ndarray, y: np.ndarray) -> float:
    """Calculation of Manhattan distance btw 2 arrays."""
    n   = x.shape[0]
    ret = 0.0
    for i in range(n):
        ret += abs(x[i] - y[i])
    return ret
