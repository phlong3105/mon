#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shape Distance and Matching functions using SciPy package.

References:
	https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
"""

from __future__ import annotations

from math import asin
from math import cos
from math import pow
from math import sin
from math import sqrt

import numpy as np
from scipy.spatial.distance import braycurtis
from scipy.spatial.distance import canberra
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import correlation
from scipy.spatial.distance import cosine
from scipy.spatial.distance import dice
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import hamming
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import kulczynski1
from scipy.spatial.distance import kulsinski
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import pdist
from scipy.spatial.distance import rogerstanimoto
from scipy.spatial.distance import russellrao
from scipy.spatial.distance import seuclidean
from scipy.spatial.distance import sokalmichener
from scipy.spatial.distance import sokalsneath
from scipy.spatial.distance import sqeuclidean
from scipy.spatial.distance import yule

from one.core import DISTANCES

__all__ = [
	"braycurtis",
	"canberra",
	"chebyshev",
	"cityblock",
	"correlation",
	"cosine",
	"dice",
	"directed_hausdorff",
	"euclidean",
	"hamming",
	"jaccard",
	"jensenshannon",
	"kulczynski1",
	"kulsinski",
	"mahalanobis",
	"minkowski",
	"pdist",
	"rogerstanimoto",
	"russellrao",
	"seuclidean",
	"sokalmichener",
	"sokalsneath",
	"sqeuclidean",
	"yule",
	
	"angle_between_vectors",
	"chebyshev_distance",
	"cosine_distance",
	"cosine_similarity",
	"euclidean_distance",
	"hausdorff_distance",
	"haversine_distance",
	"manhattan_distance",
	"pairwise_squared_distance",
	"ChebyshevDistance",
	"CosineDistance",
	"EuclideanDistance",
	"HausdorffDistance",
	"HaversineDistance",
	"ManhattanDistance",
]


# MARK: - Functional

def compute_distance(
	x,
	y,
	metric="",
	*,
	out=None,
	**kwargs
):
	pass


def angle_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate angle of 2 trajectories between two trajectories.
	"""
	vec1 = np.array([x[-1][0] - x[0][0], x[-1][1] - x[0][1]])
	vec2 = np.array([y[-1][0] - y[0][0], y[-1][1] - y[0][1]])
	L1   = np.sqrt(vec1.dot(vec1))
	L2   = np.sqrt(vec2.dot(vec2))
	
	if L1 == 0 or L2 == 0:
		return False
	
	cos   = vec1.dot(vec2) / (L1 * L2)
	angle = np.arccos(cos) * 360 / (2 * np.pi)
	
	return angle


def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Chebyshev distance is a metric defined on a vector space where the
	distance between two vectors is the greatest of their differences along any
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
	"""Compute pair-wise cosine similarity between points in `x` and `y`.
	
	References:
		https://www.codestudyblog.com/cnb2001/0119184904.html
	
	Args:
		x (np.ndarray[N, M]):
			An matrix of N samples of dimensionality M.
		y (np.ndarray[L, M]):
			An matrix of L samples of dimensionality M.
		
	Returns:
		sim (np.ndarray):
			Returns a matrix of size len(x), len(y) such that element (i, j)
            contains the squared distance between `x[i]` and `y[j]`.
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
		raise RuntimeError(f"`x` and `y` shape must be matched. "
		                   f"But got: {x.shape} != {y.shape}.")
	if x.ndim == 1:
		x = np.expand_dims(x, axis=0)
		y = np.expand_dims(y, axis=0)
	elif x.ndim != 2:
		raise RuntimeError(f"`x.ndim` must == 2. But got {x.ndim}.")
	
	x_norm = np.linalg.norm(x, axis=1, keepdims=True)
	y_norm = np.linalg.norm(y, axis=1, keepdims=True)
	return np.dot(x, y.T) / (x_norm * y_norm)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Compute pair-wise cosine distance between points in `x` and `y`.
	
	Args:
		x (np.ndarray[N, M]):
			An matrix of N samples of dimensionality M.
		y (np.ndarray[L, M]):
			An matrix of L samples of dimensionality M
	
	Returns:
		dist (np.ndarray):
			Returns a matrix of size len(x), len(y) such that element (i, j)
            contains the squared distance between `x[i]` and `y[j]`.
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
	return 1.0 - cosine_similarity(x, y)


def euclidean_distance(
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


def hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
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


def haversine_distance(x: np.ndarray, y: np.ndarray) -> float:
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
	a 		= (pow(sin(dlat / 2.0), 2.0) + cos(lat_x) * cos(lat_y) * pow(sin(dlon / 2.0), 2.0))
	return R * 2 * asin(sqrt(a))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Manhattan distance btw 2 arrays."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += abs(x[i] - y[i])
	return ret


def pairwise_squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Compute pair-wise squared distance between points in `x` and `y`.
	
	Args:
		x (np.ndarray[N, M]):
			An matrix of N samples of dimensionality M.
		y (np.ndarray[L, M]):
			An matrix of L samples of dimensionality M.
	
	Returns:
		r2 (np.ndarray):
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
	
	x2 = np.square(x).sum(axis=1)
	y2 = np.square(y).sum(axis=1)
	r2 = -2.0 * np.dot(x, y.T) + x2[:, None] + y2[None, :]
	r2 = np.clip(r2, 0.0, float(np.inf))
	return r2


# MARK: - Modules

@DISTANCES.register(name="chebyshev")
class ChebyshevDistance:
	"""Calculate Chebyshev distance."""

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return chebyshev_distance(x=x, y=y)


@DISTANCES.register(name="cosine")
class CosineDistance:
	"""Calculate Cosine distance."""

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return cosine_distance(x=x, y=y)


@DISTANCES.register(name="euclidean")
class EuclideanDistance:
	"""Calculate Euclidean distance."""

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return euclidean_distance(x=x, y=y)


@DISTANCES.register(name="hausdorff")
class HausdorffDistance:
	"""Calculate Hausdorff distance."""

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return hausdorff_distance(x=x, y=y)


@DISTANCES.register(name="haversine")
class HaversineDistance:
	"""Calculate Haversine distance."""

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return haversine_distance(x=x, y=y)


@DISTANCES.register(name="manhattan")
class ManhattanDistance:
	"""Calculate Manhattan distance."""

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return manhattan_distance(x=x, y=y)
