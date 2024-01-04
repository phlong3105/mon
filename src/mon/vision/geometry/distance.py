#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements distance functions.

In this module the array-like type can be np.ndarray, torch.Tensor, or
Sequence[Number]. We try to support both :class:`numpy.ndarray` and
:class:`torch.Tensor`.

See Also:
    https://github.com/scipy/scipy/blob/v1.10.0/scipy/spatial/distance.py
"""

from __future__ import annotations

__all__ = [
    "angle",
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
    "kulczynski1",
    "mahalanobis",
    "manhattan",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

import numpy as np
import scipy

from mon.vision import core

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Boolean Distance

def dice(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Dice dissimilarity between two boolean 1-D arrays. The Dice
    dissimilarity between :param:`u` and :param:`v`, is:
    
    .. math::
             \\frac{c_{TF} + c_{FT}}
                  {2c_{TT} + c_{FT} + c_{TF}}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n`.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Dice dissimilarity between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.dice(u=u, v=v, w=w)


def hamming(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Hamming distance between two boolean 1-D arrays. The Hamming
    distance between 1-D arrays :param:`u` and :param:`v`, is simply the
    proportion of disagreeing components in :param:`u` and :param:`v`. If
    :param:`u` and :param:`v` are boolean vectors, the Hamming distance is:
    
    .. math::
           \\frac{c_{01} + c_{10}}{n}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n`.
        
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Hamming distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.hamming(u=u, v=v, w=w)


def jaccard(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Jaccard-Needham dissimilarity between two boolean 1-D arrays.
    The Jaccard-Needham dissimilarity between 1-D arrays :param:`u` and
    :param:`v` is:

    .. math::
           \\frac{c_{TF} + c_{FT}}
                {c_{TT} + c_{FT} + c_{TF}}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Jaccard distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.jaccard(u=u, v=v, w=w)
    

def kulczynski1(u: np.ndarray, v: np.ndarray, *, w: np.ndarray | None = None) -> float:
    """Compute the Kulczynski 1 dissimilarity between two boolean 1-D arrays.
    The Kulczynski 1 dissimilarity between 1-D arrays :param:`u` and
    :param:`v` of length `n`, is defined as:

    .. math::
             \\frac{c_{11}}{c_{01} + c_{10}}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k \\in {0, 1, ..., n-1}`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Kulczynski 1 distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.kulczynski1(u=u, v=v, w=w)
    

def rogerstanimoto(u: np.ndarray, v: np.ndarray, *, w: np.ndarray | None = None) -> float:
    """Compute the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.
    The Rogers-Tanimoto dissimilarity between 1-D arrays :param:`u` and
    :param:`v` is defined as:

    .. math::
           \\frac{R}
                {c_{TT} + c_{FF} + R}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Rogers-Tanimoto distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.rogerstanimoto(u=u, v=v, w=w)
    
    
def russellrao(u: np.ndarray, v: np.ndarray, *, w: np.ndarray | None = None) -> float:
    """Compute the Russell-Rao dissimilarity between two boolean 1-D arrays.
    The Russell-Rao dissimilarity between 1-D arrays :param:`u` and
    :param:`v` is defined as:

    .. math::
          \\frac{n - c_{TT}}
               {n}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Russell-Rao distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.russellrao(u=u, v=v, w=w)
    

def sokalmichener(u: np.ndarray, v: np.ndarray, *, w: np.ndarray | None = None) -> float:
    """Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.
    The Sokal-Michener dissimilarity between 1-D arrays :param:`u` and
    :param:`v` is defined as:

    .. math::
           \\frac{R}
                {S + R}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n`, :math:`R = 2 * (c_{TF} + c_{FT})` and
        :math:`S = c_{FF} + c_{TT}`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Sokal-Michener distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.sokalmichener(u=u, v=v, w=w)
  

def sokalsneath(u: np.ndarray, v: np.ndarray, *, w: np.ndarray | None = None) -> float:
    """Compute the Sokal-Sneath dissimilarity between two boolean 1-D arrays.
    The Sokal-Sneath dissimilarity between 1-D arrays :param:`u` and :param:`v`
    is defined as:

    .. math::
           \\frac{R}
                {c_{TT} + R}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n` and :math:`R = 2(c_{TF} + c_{FT})`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Sokal-Sneath distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.sokalsneath(u=u, v=v, w=w)
    

def yule(u: np.ndarray, v: np.ndarray, *, w: np.ndarray | None = None) -> float:
    """Compute the Yule dissimilarity between two boolean 1-D arrays.
    The Yule dissimilarity between 1-D arrays :param:`u` and :param:`v`
    is defined as:

    .. math::
             \\frac{R}{c_{TT} * c_{FF} + \\frac{R}{2}}
    where :math:`c_{ij}` is the number of occurrences of
        :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
        :math:`k < n` and :math:`R = 2.0 * c_{TF} * c_{FT}`.

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Yule distance between vectors :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.yule(u=u, v=v, w=w)
   
   
# endregion


# region Numeric Distance

def braycurtis(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Bray-Curtis distance between two 1-D arrays. The Bray-Curtis
    distance between :param:`u` and :param:`v` is defined as:
    
    .. math::
             \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}
    The Bray-Curtis distance is in the range [0, 1] if all coordinates are
    positive, and is undefined if the inputs are of length zero.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Bray-Curtis distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.braycurtis(u=u, v=v, w=w)


def canberra(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Canberra distance between two 1-D arrays. The Canberra
    distance between :param:`u` and :param:`v` is defined as:
    
    .. math::
             d(u,v) = \\sum_i \\frac{|u_i-v_i|}{|u_i|+|v_i|}.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Canberra distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.canberra(u=u, v=v, w=w)
    
    
def chebyshev(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Chebyshev distance between two 1-D arrays. The Chebyshev
    distance between :param:`u` and :param:`v` is defined as:
    
    .. math::
             \\max_i {|u_i-v_i|}.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Chebyshev distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.chebyshev(u=u, v=v, w=w)
    

def cityblock(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the City Block (Manhattan) distance between two 1-D arrays. The
    Manhattan distance between :param:`u` and :param:`v` is defined as:
    
    .. math::
             \\sum_i {\\left| u_i - v_i \\right|}.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The City Block (Manhattan) distance between 1-D arrays :param:`u` and
        :param:`v`.
    """
    return scipy.spatial.distance.cityblock(u=u, v=v, w=w)
    

def correlation(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the correlation distance between two 1-D arrays. The correlation
    distance between :param:`u` and :param:`v` is defined as:
    
    .. math::
             1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
             {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The correlation distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.correlation(u=u, v=v, w=w)
    

def cosine(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Cosine distance between two 1-D arrays. The Cosine distance
    between :param:`u` and :param:`v` is defined as:
    
    .. math::
             1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}.
    
    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Cosine distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.cosine(u=u, v=v, w=w)
    

def euclidean(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the Euclidean distance between two 1-D arrays. The Euclidean
    distance between :param:`u` and :param:`v` is defined as:

    .. math::
             {\\|u-v\\|}_2 \\left(\\sum{(w_i |(u_i - v_i)|^2)}\\right)^{1/2}

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The Euclidean distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.euclidean(u=u, v=v, w=w)
    

def mahalanobis(u: np.ndarray, v: np.ndarray, vi: np.ndarray) -> float:
    """Compute the Mahalanobis distance between two 1-D arrays. The Mahalanobis
    distance between :param:`u` and :param:`v` is defined as:

    .. math::
              \\sqrt{ (u-v) V^{-1} (u-v)^T }
    where `V` is the covariance matrix. Note that the argument `VI` is the
    inverse of `V`.

    Args:
        u: An array-like input.
        v: An array-like input.
        vi: The inverse of the covariance matrix.

    Returns:
        The Mahalanobis distance between 1-D arrays :param:`u` and :param:`v`.
    """
    return scipy.spatial.distance.mahalanobis(u=u, v=v, VI=vi)


manhattan = cityblock


def minkowski(
    u: np.ndarray,
    v: np.ndarray,
    o: int        = 2,
    w: np.ndarray = None
) -> float:
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
    return scipy.spatial.distance.minkowski(u=u, v=v, p=o, w=w)


def seuclidean(u: np.ndarray, v: np.ndarray, V: np.ndarray) -> float:
    """Compute the standardized Euclidean distance between two 1-D arrays. The
    standardized Euclidean distance between :param:`u` and :param:`v` is defined
    as:

    .. math::
             {\\|u-v\\|}_2 \\left(\\sum{(w_i |(u_i - v_i)|^2)}\\right)^{1/2}

    Args:
        u: An array-like input.
        v: An array-like input.
        V: an 1-D array of component variances. It is usually computed among a
            larger collection of vectors.

    Returns:
        The standardized Euclidean distance between 1-D arrays :param:`u` and
        :param:`v`.
    """
    return scipy.spatial.distance.seuclidean(u=u, v=v, V=V)
    

def sqeuclidean(u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Compute the squared Euclidean distance between two 1-D arrays. The
    squared Euclidean distance between :param:`u` and :param:`v` is defined as:

    .. math::
             {\\|u-v\\|}_2 \\left(\\sum{(w_i |(u_i - v_i)|^2)}\\right)^{1/2}

    Args:
        u: An array-like input.
        v: An array-like input.
        w: The weights for each value in :param:`u` and :param:`v`. Default to
            None, which gives each value a weight of 1.0.

    Returns:
        The squared Euclidean distance between 1-D arrays :param:`u` and
        :param:`v`.
    """
    return scipy.spatial.distance.sqeuclidean(u=u, v=v, w=w)
    

# endregion


# region Directed Vector (Geometry) Distance

def angle(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate the angle distance between two directed vectors (geometry).
    
    Args:
        u: An array-like input.
        v: An array-like input.
        
    Returns:
        The angle distance between directed vectors :param:`u` and :param:`v`.
    """
    vec1 = np.array([u[-1][0] - u[0][0], u[-1][1] - u[0][1]])
    vec2 = np.array([v[-1][0] - v[0][0], v[-1][1] - v[0][1]])
    l1   = np.sqrt(vec1.dot(vec1))
    l2   = np.sqrt(vec2.dot(vec2))
    if l1 == 0 or l2 == 0:
        return False
    cos   = vec1.dot(vec2) / (l1 * l2)
    angle = np.arccos(cos) * 360 / (2 * np.pi)
    return angle


# endregion


# region Matrix Distance

def directed_hausdorff(
    u: np.ndarray,
    v: np.ndarray,
    seed: int = 0
) -> tuple[float, int, int]:
    """Compute the directed Hausdorff distance between two 2-D arrays. Distances
    between pairs are calculated using a Euclidean metric.
    
    Args:
        u: An :math:`[M, N]` array-like input.
        v: An :math:`[M, N]` array-like input.
        seed: Default: ``0``, a random shuffling of :param:`u` and :param:`v`
            that guarantees reproducibility.

    Returns:
        The directed Hausdorff distance between arrays :param:`u` and
        :param:`v`.
        The index of the point contributing to the Hausdorff pair in :param:`u`.
        The index of the point contributing to the Hausdorff pair in :param:`v`.
        
    Examples:
        Find the directed Hausdorff distance between two 2-D arrays of
        coordinates:
        >>> from mon.vision.geometry.distance import directed_hausdorff
        >>> import numpy as np
        >>> u = np.array([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
        >>> v = np.array([(2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -4.0)])
        >>> directed_hausdorff(u, v)[0]
        2.23606797749979
        >>> directed_hausdorff(v, u)[0]
        3.0
        Find the general (symmetric) Hausdorff distance between two 2-D
        arrays of coordinates:
        >>> max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        3.0
        Find the indices of the points that generate the Hausdorff distance
        (the Hausdorff pair):
        >>> directed_hausdorff(v, u)[1:]
        (3, 3)
    """
    return scipy.spatial.distance.directed_hausdorff(u=u, v=v, seed=seed)


# endregion
