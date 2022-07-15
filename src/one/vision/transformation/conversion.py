#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import enum
import inspect
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from one.core import error_console
from one.core import eye_like
from one.core import inverse_cast
from one.core import PI
from one.core import solve_cast


class QuaternionCoeffOrder(enum.Enum):
    XYZW = "xyzw"
    WXYZ = "wxyz"


# MARK: - Rad <-> Deg

def rad_to_deg(tensor: Tensor) -> Tensor:
    r"""Function that converts angles from radians to degrees.

    Args:
        tensor (Tensor):
            Tensor of arbitrary shape.

    Returns:
        (Tensor):
            Tensor with same shape as input.

    Example:
        >>> input = torch.tensor(3.1415926535)
        >>> rad_to_deg(input)
        image(180.)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"`tensor` must be `Tensor`. But got: {type(tensor)}.")

    return 180.0 * tensor / PI.to(tensor.device).type(tensor.dtype)


def deg_to_rad(tensor: Tensor) -> Tensor:
    r"""Function that converts angles from degrees to radians.

    Args:
        tensor (Tensor):
            Tensor of arbitrary shape.

    Returns:
        (Tensor):
            Tensor with same shape as input.

    Examples:
        >>> input = torch.tensor(180.)
        >>> deg_to_rad(input)
        image(3.1416)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"`tensor` must be `Tensor`. But got: {type(tensor)}.")

    return tensor * PI.to(tensor.device).type(tensor.dtype) / 180.0


# MARK: - Polar <-> Cartesian

def polar_to_cart(rho: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
    r"""Function that converts polar coordinates to cartesian coordinates.

    Args:
        rho (Tensor):
            Tensor of arbitrary shape.
        phi (Tensor):
            Tensor of same arbitrary shape.

    Returns:
        (Tensor):
            Tensor with same shape as input.

    Example:
        >>> rho = torch.rand(1, 3, 3)
        >>> phi = torch.rand(1, 3, 3)
        >>> x, y = polar_to_cart(rho, phi)
    """
    if not (isinstance(rho, Tensor) & isinstance(phi, Tensor)):
        raise TypeError(f"`rho` and `phi` must be a `Tensor`. "
                        f"But got: {type(rho)}, {type(phi)}.")

    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def cart_to_polar(
	x: Tensor, y: Tensor, eps: float = 1.0e-8
) -> tuple[Tensor, Tensor]:
    """Function that converts cartesian coordinates to polar coordinates.

    Args:
        x (Tensor):
            Tensor of arbitrary shape.
        y (Tensor):
            Tensor of same arbitrary shape.
        eps (float):
            To avoid division by zero.

    Example:
        >>> x = torch.rand(1, 3, 3)
        >>> y = torch.rand(1, 3, 3)
        >>> rho, phi = cart_to_polar(x, y)
    """
    if not (isinstance(x, Tensor) & isinstance(y, Tensor)):
        raise TypeError(f"`x` and `y` must be a `Tensor`. "
                        f"But got: {type(x)}, {type(y)}.")

    rho = torch.sqrt(x ** 2 + y ** 2 + eps)
    phi = torch.atan2(y, x)
    return rho, phi


# MARK: - Point <-> Homogeneous

def convert_points_from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points (Tensor):
            Points to be transformed of shape [B, N, D].
        eps (float):
            To avoid division by zero.

    Returns:
        (Tensor):
            Points in Euclidean space [B, N, D-1].

    Examples:
        >>> input = torch.tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        image([[0., 0.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"`points` must be a `Tensor`. But got: {type(points)}.")

    if points.ndim < 2:
        raise ValueError(f"`points.ndim` must >= 2. But got: {points.shape}.")

    # We check for points at max_val
    z_vec = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask  = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: Tensor) -> Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points (Tensor):
            Points to be transformed with shape [B, N, D].

    Returns:
        (Tensor):
            Points in homogeneous coordinates [B, N, D+1].

    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        image([[0., 0., 1.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"`points` must be a `Tensor`. But got: {type(points)}.")
    if len(points.shape) < 2:
        raise ValueError(f"`points.ndim` must >= 2. But got: {points.shape}.")

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


# MARK: - Affine Matrix <-> Homography

def _convert_affine_matrix_to_homography(A: Tensor) -> Tensor:
    H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def convert_affine_matrix_to_homography(A: Tensor) -> Tensor:
    r"""Function that converts batch of affine matrices.

    Args:
        A (Tensor):
            Affine matrix with shape [B, 2, 3].

    Returns:
         (Tensor):
            Homography matrix with shape of [B, 3, 3].

    Examples:
        >>> A = torch.tensor([[[1., 0., 0.],
        ...                    [0., 1., 0.]]])
        >>> convert_affine_matrix_to_homography(A)
        image([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]])
    """
    if not isinstance(A, Tensor):
        raise TypeError(f"`A` must be a `Tensor`. But got: {type(A)}.")

    if not (A.ndim == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError(f"`A` must have the shape of [B, 2, 3]. But got: {A.shape}.")

    return _convert_affine_matrix_to_homography(A)


def convert_affine_matrix_to_homography3d(A: Tensor) -> Tensor:
    r"""Function that converts batch of 3d affine matrices.

    Args:
        A (Tensor):
            Affine matrix with shape [B, 3, 4].

    Returns:
         (Tensor):
            Homography matrix with shape of [B, 4, 4].

    Examples:
        >>> A = torch.tensor([[[1., 0., 0., 0.],
        ...                    [0., 1., 0., 0.],
        ...                    [0., 0., 1., 0.]]])
        >>> convert_affine_matrix_to_homography3d(A)
        image([[[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., 1.]]])
    """
    if not isinstance(A, Tensor):
        raise TypeError(f"`A` must be a `Tensor.` But got: {type(A)}.")

    if not (A.ndim == 3 and A.shape[-2:] == (3, 4)):
        raise ValueError(f"`A` must have the shape of [B, 3, 4]. But got: {A.shape}.")

    return _convert_affine_matrix_to_homography(A)


# MARK: - Angle Axis <-> Rotation Matrix

def angle_axis_to_rotation_matrix(angle_axis: Tensor) -> Tensor:
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.

    Args:
        angle_axis (Tensor):
            Tensor of 3d vector of axis-angle rotations in radians with shape
            [N, 3].

    Returns:
        (Tensor):
            Tensor of rotation matrices of shape [N, 3, 3].

    Example:
        >>> input = torch.tensor([[0., 0., 0.]])
        >>> angle_axis_to_rotation_matrix(input)
        image([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]])

        >>> input = torch.tensor([[1.5708, 0., 0.]])
        >>> angle_axis_to_rotation_matrix(input)
        image([[[ 1.0000e+00,  0.0000e+00,  0.0000e+00],
                 [ 0.0000e+00, -3.6200e-06, -1.0000e+00],
                 [ 0.0000e+00,  1.0000e+00, -3.6200e-06]]])
    """
    if not isinstance(angle_axis, Tensor):
        raise TypeError(f"`angle_axis` must be a `Tensor`. But got: {type(angle_axis)}.")

    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"`angle_axis` must have the shape of [*, 3]. "
                         f"But got: {angle_axis.shape}.")

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the norm of
        # the angle_axis vector is greater than zero. Otherwise, we get a
        # division by zero.
        k_one      = 1.0
        theta      = torch.sqrt(theta2)
        wxyz       = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta  = torch.cos(theta)
        sin_theta  = torch.sin(theta)

        r00             = cos_theta + wx * wx * (k_one - cos_theta)
        r10             = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20             = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01             = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11             = cos_theta + wy * wy * (k_one - cos_theta)
        r21             = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02             = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12             = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22             = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz      = torch.chunk(angle_axis, 3, dim=1)
        k_one           = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2      = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2      = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps      = 1e-6
    mask     = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    # create output pose matrix
    batch_size      = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = (mask_pos * rotation_matrix_normal
                                    + mask_neg * rotation_matrix_taylor)
    return rotation_matrix  # Nx3x3


def rotation_matrix_to_angle_axis(rotation_matrix: Tensor) -> Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector in radians.

    Args:
        rotation_matrix (Tensor):
            Rotation matrix of shape [N, 3, 3].

    Returns:
        (Tensor):
            Rodrigues vector transformation of shape [N, 3].

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_angle_axis(input)
        image([0., 0., 0.])

        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 0., -1.],
        ...                       [0., 1., 0.]])
        >>> rotation_matrix_to_angle_axis(input)
        image([1.5708, 0.0000, 0.0000])
    """
    if not isinstance(rotation_matrix, Tensor):
        raise TypeError(f"`rotation_matrix` must be a `Tensor`. "
                        f"But got: {type(rotation_matrix)}.")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"`rotation_matrix` must have the shape of [*, 3, 3]. "
                         f"But got: {rotation_matrix.shape}.")
    quaternion = rotation_matrix_to_quaternion(
	    rotation_matrix, order=QuaternionCoeffOrder.WXYZ
    )
    return quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)


# MARK: - Rotation Matrix <-> Quaternion

def rotation_matrix_to_quaternion(
    rotation_matrix: Tensor,
	eps            : float                = 1.0e-8,
	order          : QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in [w, x, y, z] or [x, y, z, w] format.

    .. note::
        The [x, y, z, w] order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix (Tensor):
            Rotation matrix to convert with shape [*, 3, 3].
        eps (float):
            Small value to avoid zero division.
        order (QuaternionCoeffOrder):
            Quaternion coefficient order. Note: 'xyzw' will be deprecated in
            favor of 'wxyz'.

    Return:
        (Tensor):
            Rotation in quaternion with shape [*, 4].

    Example:
        >>> input = torch.tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps,
        ...                               order=QuaternionCoeffOrder.WXYZ)
        image([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(rotation_matrix)}.")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a [*, 3, 3] image."
                         f" Got: {rotation_matrix.shape}.")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of "
                             f"{QuaternionCoeffOrder.__members__.keys()}.")

    if order == QuaternionCoeffOrder.XYZW:
        error_console.log(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    def safe_zero_division(numerator: Tensor, denominator: Tensor) -> Tensor:
        eps = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
	    rotation_matrix_vec, chunks=9, dim=-1
    )
    trace = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(quaternion: Tensor, eps: float = 1.0e-12) -> Tensor:
    r"""Normalize a quaternion.

    The quaternion should be in [x, y, z, w] or [w, x, y, z] format.

    Args:
        quaternion (Tensor):
            Tensor containing a quaternion to be normalized. he image can be
            of shape [*, 4].
        eps (float):
            Small value to avoid division by zero.

    Return:
        (Tensor):
            Normalized quaternion of shape [*, 4].

    Example:
        >>> quaternion = torch.tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        image([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. "
                        f"Got: {type(quaternion)}.")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a image of shape [*, 4]. "
                         f"Got: {quaternion.shape}.")
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(
    quaternion: Tensor,
    order     : QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> Tensor:
    r"""Convert a quaternion to a rotation matrix.

    The quaternion should be in [x, y, z, w] or [w, x, y, z] format.
	
	Based on:
	https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
	https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247

    Args:
        quaternion (Tensor):
            Tensor containing a quaternion to be converted. The image can be of
            shape [*, 4].
        order (QuaternionCoeffOrder):
            Quaternion coefficient order. Note: 'xyzw' will be deprecated in
            favor of 'wxyz'.

    Return:
        (Tensor):
            Rotation matrix of shape [*, 3, 3].

    Example:
        >>> quaternion = torch.tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        image([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(quaternion)}.")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a image of shape [*, 4]. "
                         f"Got: {quaternion.shape}.")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"Order must be one of "
                             f"{QuaternionCoeffOrder.__members__.keys()}.")

    if order == QuaternionCoeffOrder.XYZW:
        error_console.log(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # normalize the input quaternion
    quaternion_norm = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    if order == QuaternionCoeffOrder.XYZW:
        x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)
    else:
        w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx  = 2.0 * x
    ty  = 2.0 * y
    tz  = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


# MARK: - Angle Axis <-> Quaternion

def quaternion_to_angle_axis(
    quaternion: Tensor,
    order     : QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> Tensor:
    """Convert quaternion vector to angle axis of rotation in radians.

    The quaternion should be in [x, y, z, w] or [w, x, y, z] format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (Tensor):
            Tensor with quaternions.
        order (QuaternionCoeffOrder):
            Quaternion coefficient order. Note: 'xyzw' will be deprecated in
            favor of 'wxyz'.

    Return:
        (Tensor):
            Tensor with angle axis of rotation.

    Shape:
        - Input: [*, 4] where `*` means, any number of dimensions
        - Output: [*, 3]

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_to_angle_axis(quaternion)
        image([3.1416, 0.0000, 0.0000])
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(f"Input type is not a Tensor. Got: {type(quaternion)}.")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a image of shape [N, 4] or 4. "
                         f"Got: {quaternion.shape}.")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of "
                             f"{QuaternionCoeffOrder.__members__.keys()}.")

    if order == QuaternionCoeffOrder.XYZW:
        error_console.log(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )
    # unpack input and compute conversion
    q1        = torch.tensor([])
    q2        = torch.tensor([])
    q3        = torch.tensor([])
    cos_theta = torch.tensor([])

    if order == QuaternionCoeffOrder.XYZW:
        q1        = quaternion[..., 0]
        q2        = quaternion[..., 1]
        q3        = quaternion[..., 2]
        cos_theta = quaternion[..., 3]
    else:
        cos_theta = quaternion[..., 0]
        q1        = quaternion[..., 1]
        q2        = quaternion[..., 2]
        q3        = quaternion[..., 3]

    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k     = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis          = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def angle_axis_to_quaternion(
    angle_axis: Tensor,
    order     : QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> Tensor:
    r"""Convert an angle axis to a quaternion.

    The quaternion vector has components in [x, y, z, w] or [w, x, y, z] format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
	
	Based on: https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138
    
    Args:
        angle_axis (Tensor):
            Tensor with angle axis in radians.
        order (QuaternionCoeffOrder):
            Quaternion coefficient order. Note: 'xyzw' will be deprecated in
            favor of 'wxyz'.

    Return:
        (Tensor):
            Tensor with quaternion.

    Shape:
        - Input: [*, 3] where `*` means, any number of dimensions
        - Output: [*, 4]

    Example:
        >>> angle_axis = torch.tensor((0., 1., 0.))
        >>> angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)
        image([0.8776, 0.0000, 0.4794, 0.0000])
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError(f"Input type is not a Tensor. Got: {type(angle_axis)}.")

    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"Input must be a image of shape [N, 3] or 3. "
                         f"Got: {angle_axis.shape}.")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of "
                             f"{QuaternionCoeffOrder.__members__.keys()}.")

    if order == QuaternionCoeffOrder.XYZW:
        error_console.log(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # unpack input and compute conversion
    a0            = angle_axis[..., 0:1]
    a1            = angle_axis[..., 1:2]
    a2            = angle_axis[..., 2:3]
    theta_squared = a0 * a0 + a1 * a1 + a2 * a2

    theta      = torch.sqrt(theta_squared)
    half_theta = theta * 0.5

    mask = theta_squared > 0.0
    ones = torch.ones_like(half_theta)

    k_neg = 0.5 * ones
    k_pos = torch.sin(half_theta) / theta
    k     = torch.where(mask, k_pos, k_neg)
    w     = torch.where(mask, torch.cos(half_theta), ones)

    quaternion = torch.zeros(
        size   = (*angle_axis.shape[:-1], 4),
	    dtype  = angle_axis.dtype,
	    device = angle_axis.device
    )
    if order == QuaternionCoeffOrder.XYZW:
        quaternion[..., 0:1] = a0 * k
        quaternion[..., 1:2] = a1 * k
        quaternion[..., 2:3] = a2 * k
        quaternion[..., 3:4] = w
    else:
        quaternion[..., 1:2] = a0 * k
        quaternion[..., 2:3] = a1 * k
        quaternion[..., 3:4] = a2 * k
        quaternion[..., 0:1] = w
    return quaternion


# MARK: - Quaternion Log <-> Quaternion Exp

def quaternion_log_to_exp(
    quaternion: Tensor,
	eps       : float                = 1.0e-8,
	order     : QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> Tensor:
    r"""Apply exponential map to log quaternion.

    The quaternion should be in [x, y, z, w] or [w, x, y, z] format.

    Args:
        quaternion (Tensor):
            Tensor containing a quaternion to be converted. The tensor can be
            of shape [*, 3].
        eps (float):
            Small number for clamping.
        order (QuaternionCoeffOrder):
            Quaternion coefficient order. Note: 'xyzw' will be deprecated in
            favor of 'wxyz'.

    Return:
        (Tensor):
            Quaternion exponential map of shape [*, 4].

    Example:
        >>> quaternion = torch.tensor((0., 0., 0.))
        >>> quaternion_log_to_exp(quaternion, eps=torch.finfo(quaternion.dtype).eps,
        ...                       order=QuaternionCoeffOrder.WXYZ)
        image([1., 0., 0., 0.])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(quaternion)}.")

    if not quaternion.shape[-1] == 3:
        raise ValueError(f"Input must be a image of shape [*, 3]. "
                         f"Got: {quaternion.shape}.")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of "
                             f"{QuaternionCoeffOrder.__members__.keys()}.")

    if order == QuaternionCoeffOrder.XYZW:
        error_console.log(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # compute quaternion norm
    norm_q = torch.norm(quaternion, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # compute scalar and vector
    quaternion_vector = quaternion * torch.sin(norm_q) / norm_q
    quaternion_scalar = torch.cos(norm_q)

    # compose quaternion and return
    quaternion_exp = torch.tensor([])
    if order == QuaternionCoeffOrder.XYZW:
        quaternion_exp = torch.cat((quaternion_vector, quaternion_scalar), dim=-1)
    else:
        quaternion_exp = torch.cat((quaternion_scalar, quaternion_vector), dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(
    quaternion: Tensor,
	eps       : float                = 1.0e-8,
	order     : QuaternionCoeffOrder = QuaternionCoeffOrder.XYZW
) -> Tensor:
    r"""Apply the log map to a quaternion.

    The quaternion should be in [x, y, z, w] format.

    Args:
        quaternion (Tensor):
            Tensor containing a quaternion to be converted. The tensor can be of
            shape [*, 4].
        eps (float):
            Small number for clamping.
        order (QuaternionCoeffOrder):
            Quaternion coefficient order. Note: 'xyzw' will be deprecated in
            favor of 'wxyz'.

    Return:
        (Tensor):
            Quaternion log map of shape [*, 3].

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=torch.finfo(quaternion.dtype).eps,
        ...                       order=QuaternionCoeffOrder.WXYZ)
        image([0., 0., 0.])
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(quaternion)}.")

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a image of shape [*, 4]. "
                         f"Got: {quaternion.shape}.")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of "
                             f"{QuaternionCoeffOrder.__members__.keys()}.")

    if order == QuaternionCoeffOrder.XYZW:
        error_console.log(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # unpack quaternion vector and scalar
    quaternion_vector = torch.tensor([])
    quaternion_scalar = torch.tensor([])

    if order == QuaternionCoeffOrder.XYZW:
        quaternion_vector = quaternion[..., 0:3]
        quaternion_scalar = quaternion[..., 3:4]
    else:
        quaternion_scalar = quaternion[..., 0:1]
        quaternion_vector = quaternion[..., 1:4]

    # compute quaternion norm
    norm_q = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # apply log map
    quaternion_log = (
        quaternion_vector * torch.acos(torch.clamp(quaternion_scalar, min=-1.0, max=1.0)) / norm_q
    )

    return quaternion_log


# MARK: - Normalize Pixel

def normalize_pixel_coordinates(
    pixel_coordinates: Tensor, height: int, width: int, eps: float = 1e-8
) -> Tensor:
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).
	
	Based on:
	https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L65-L71
    
    Args:
        pixel_coordinates (Tensor):
            Grid with pixel coordinates. Shape can be [*, 2].
        width (int):
            Maximum width in the x-axis.
        height (int):
            Maximum height in the y-axis.
        eps (float):
            Safe division by zero.

    Return:
        (Tensor):
            Normalized pixel coordinates with shape [*, 2].

    Examples:
        >>> coords = torch.tensor([[50., 100.]])
        >>> normalize_pixel_coordinates(coords, 100, 50)
        image([[1.0408, 1.0202]])
    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape [*, 2]. "
                         f"Got: {pixel_coordinates.shape}.")

    # compute normalization factor
    hw = torch.stack(
        [
            torch.tensor(width,  device=pixel_coordinates.device, dtype=pixel_coordinates.dtype),
            torch.tensor(height, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype),
        ]
    )

    factor = torch.tensor(
	    2.0, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype
    ) / (hw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates(
    pixel_coordinates: Tensor, height: int, width: int, eps: float = 1e-8
) -> Tensor:
    r"""Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right
    (x = w-1).

    Args:
        pixel_coordinates (Tensor):
            Normalized grid coordinates. Shape can be [*, 2].
        width (int):
            Maximum width in the x-axis.
        height (int):
            Maximum height in the y-axis.
        eps (float):
            Safe division by zero.

    Return:
        (Tensor):
            Denormalized pixel coordinates with shape [*, 2].

    Examples:
        >>> coords = torch.tensor([[-1., -1.]])
        >>> denormalize_pixel_coordinates(coords, 100, 50)
        image([[0., 0.]])
    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape [*, 2]. "
                         f"Got: {pixel_coordinates.shape}")
    # compute normalization factor
    hw = (
        torch.stack([torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor = torch.tensor(2.0) / (hw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


def normalize_pixel_coordinates3d(
    pixel_coordinates: Tensor,
	depth            : int,
	height           : int,
	width            : int,
	eps              : float = 1e-8
) -> Tensor:
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates (Tensor):
            Grid with pixel coordinates. Shape can be [*, 3].
        depth (int):
            Maximum depth in the z-axis.
        height (int):
            Maximum height in the y-axis.
        width (int):
            Maximum width in the x-axis.
        eps (float):
            Safe division by zero.

    Return:
        (Tensor):
            Normalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(f"Input pixel_coordinates must be of shape [*, 3]. "
                         f"Got: {pixel_coordinates.shape}")
    # compute normalization factor
    dhw = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates3d(
    pixel_coordinates: Tensor,
	depth            : int,
	height           : int,
	width            : int,
	eps              : float  = 1e-8
) -> Tensor:
    r"""Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right
    (x = w-1).

    Args:
        pixel_coordinates (Tensor):
            Normalized grid coordinates. Shape can be [*, 3].
        depth (int):
            Maximum depth in the x-axis.
        height (int):
            Maximum height in the y-axis.
        width (int):
            Maximum width in the x-axis.
        eps (float):
            Safe division by zero.

    Return:
        (Tensor):
            Denormalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(f"Input pixel_coordinates must be of shape [*, 3]. "
                         f"Got: {pixel_coordinates.shape}.")
    # compute normalization factor
    dhw = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


# MARK: - Angle <-> Rotation Matrix

def angle_to_rotation_matrix(angle: Tensor) -> Tensor:
    r"""Create a rotation matrix out of angles in degrees.

    Args:
        angle (Tensor):
            Angles in degrees, any shape [*].

    Returns:
        (Tensor):
            Rotation matrices with shape [*, 2, 2].

    Example:
        >>> input  = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = deg_to_rad(angle)
    cos_a   = torch.cos(ang_rad)
    sin_a   = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


# MARK: - Get Transform

def _build_perspective_param(p: Tensor, q: Tensor, axis: str) -> Tensor:
    ones  = torch.ones_like(p)[..., 0:1]
    zeros = torch.zeros_like(p)[..., 0:1]
    if axis == "x":
        return torch.cat(
            [
                p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
                -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
            ], dim=1
        )

    if axis == "y":
        return torch.cat(
            [
                zeros, zeros, zeros,  p[:, 0:1], p[:, 1:2], ones,
                -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]
            ], dim=1
        )

    raise NotImplementedError(f"perspective params for axis `{axis}` is not "
                              f"implemented.")


def get_perspective_transform(src, dst):
    r"""Calculate a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where
    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src:
            Coordinates of quadrangle vertices in the source image with shape
            [B, 4, 2].
        dst:
            Coordinates of the corresponding quadrangle vertices in the
            destination image with shape [B, 4, 2].

    Returns:
        Perspective transformation with shape [B, 3, 3].

    .. note::
        This function is often used in conjunction with :func:`warp_perspective`.
    """
    if not isinstance(src, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(src)}")

    if not isinstance(dst, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got: {type(dst)}")

    if not src.dtype == dst.dtype:
        raise TypeError(f"Source data type {src.dtype} must match Destination "
                        f"data type {dst.dtype}")

    if not src.shape[-2:] == (4, 2):
        raise ValueError(f"Inputs must be a [B, 4, 2] image. Got: {src.shape}")

    if not src.shape == dst.shape:
        raise ValueError(f"Inputs must have the same shape. Got: {dst.shape}")

    if not (src.shape[0] == dst.shape[0]):
        raise ValueError(f"Inputs must have same batch size dimension. "
                         f"Expect {src.shape} but got {dst.shape}")

    # We build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here we could even pass
    # more correspondence
    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack(
        [
            dst[:, 0:1, 0],
            dst[:, 0:1, 1],
            dst[:, 1:2, 0],
            dst[:, 1:2, 1],
            dst[:, 2:3, 0],
            dst[:, 2:3, 1],
            dst[:, 3:4, 0],
            dst[:, 3:4, 1],
        ],
        dim=1,
    )

    # solve the system Ax = b
    X, _ = solve_cast(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M          = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)

    return M.view(-1, 3, 3)  # Bx3x3


def projection_from_Rt(rmat: Tensor, tvec: Tensor) -> Tensor:
    r"""Compute the projection matrix from Rotation and translation.

    .. warning::
        This API signature it is experimental and might suffer some changes in
        the future.

    Concatenates the batch of rotations and translations such that
    :math:`P = [R | t]`.

    Args:
       rmat (Tensor): 
            Rotation matrix with shape [*, 3, 3].
       tvec (Tensor):
            Translation vector with shape [*, 3, 1].

    Returns:
        (Tensor):
            Projection matrix with shape [*, 3, 4].
    """
    if not (len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3)):
        raise AssertionError(rmat.shape)
    if not (len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1)):
        raise AssertionError(tvec.shape)

    return torch.cat([rmat, tvec], dim=-1)  # Bx3x4


def get_projective_transform(
    center: Tensor, angles: Tensor, scales: Tensor
) -> Tensor:
    r"""Calculate the projection matrix for a 3D rotation.

    .. warning::
        This API signature it is experimental and might suffer some changes in
        the future.

    The function computes the projection matrix given the center and angles per
    axis.

    Args:
        center (Tensor):
            Center of the rotation [x, y, z] in the source with shape [B, 3].
        angles (Tensor):
            Angle axis vector containing the rotation angles in degrees in the
            form of [rx, ry, rz] with shape [B, 3]. Internally it calls
            Rodrigues to compute the rotation matrix from axis-angle.
        scales (Tensor):
            Scale factor for x-y-z-directions with shape [B, 3].

    Returns:
        (Tensor):
            Projection matrix of 3D rotation with shape [B, 3, 4].

    .. note::
        This function is often used in conjunction with :func:`warp_affine3d`.
    """
    if not (len(center.shape) == 2 and center.shape[-1] == 3):
        raise AssertionError(center.shape)
    if not (len(angles.shape) == 2 and angles.shape[-1] == 3):
        raise AssertionError(angles.shape)
    if center.device != angles.device:
        raise AssertionError(center.device, angles.device)
    if center.dtype != angles.dtype:
        raise AssertionError(center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad = deg_to_rad(angles)
    rmat           = angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3
    scaling_matrix = eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat           = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = torch.eye(4)[None].repeat(rmat.shape[0], 1, 1).type_as(center)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = inverse_cast(from_origin_mat)

    # append translation with zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affine_matrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = from_origin_mat @ proj_mat @ to_origin_mat

    return proj_mat[..., :3, :]  # Bx3x4


# MARK: - Get Matrix

def get_affine_matrix2d(
    translations: Tensor,
    center      : Tensor,
    scale       : Tensor,
    angle       : Tensor,
    sx          : Optional[Tensor] = None,
    sy          : Optional[Tensor] = None,
) -> Tensor:
    r"""Compose affine matrix from the components.

    Args:
        translations (Tensor):
            Image containing the translation vector with shape [B, 2].
        center (Tensor):
            Image containing the center vector with shape [B, 2].
        scale (Tensor):
            Image containing the scale factor with shape [B, 2].
        angle (Tensor):
            Image of angles in degrees [B].
        sx (Tensor, optional):
            Image containing the shear factor in the x-direction with shape [B].
        sy (Tensor, optional):
            Image containing the shear factor in the y-direction with shape [B].

    Returns:
         (Tensor):
            The affine transformation matrix [B, 3, 3].

    .. note::
        This function is often used in conjunction with :func:`warp_affine`,
        :func:`warp_perspective`.
    """
    transform          = get_rotation_matrix2d(center, -angle, scale)
    transform[..., 2] += translations  # tx/ty

    # pad transform to get Bx3x3
    transform_h = convert_affine_matrix_to_homography(transform)

    if any(s is not None for s in [sx, sy]):
        shear_mat   = get_shear_matrix2d(center, sx, sy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_affine_matrix3d(
    translations: Tensor,
    center      : Tensor,
    scale       : Tensor,
    angles      : Tensor,
    sxy         : Optional[Tensor] = None,
    sxz         : Optional[Tensor] = None,
    syx         : Optional[Tensor] = None,
    syz         : Optional[Tensor] = None,
    szx         : Optional[Tensor] = None,
    szy         : Optional[Tensor] = None,
) -> Tensor:
    r"""Compose 3d affine matrix from the components.

    Args:
        translations (Tensor):
            Image containing the translation vector (dx,dy,dz) with shape [B, 3].
        center (Tensor):
            Image containing the center vector (x,y,z) with shape [B, 3].
        scale (Tensor):
            Image containing the scale factor with shape [B].
        angles (Tensor):
            Angle axis vector containing the rotation angles in degrees in the
            form of [rx, ry, rz] with shape [B, 3]. Internally it calls
            Rodrigues to compute the rotation matrix from axis-angle.
        sxy (Tensor, optional):
            Image containing the shear factor in the xy-direction with shape [B].
        sxz (Tensor, optional):
            Image containing the shear factor in the xz-direction with shape [B].
        syx (Tensor, optional):
            Image containing the shear factor in the yx-direction with shape [B].
        syz (Tensor, optional):
            Image containing the shear factor in the yz-direction with shape [B].
        szx (Tensor, optional):
            Image containing the shear factor in the zx-direction with shape [B].
        szy (Tensor, optional):
            Image containing the shear factor in the zy-direction with shape [B].

    Returns:
        (Tensor):
            The 3d affine transformation matrix [B, 3, 3].

    .. note::
        This function is often used in conjunction with :func:`warp_perspective`.
    """
    transform          = get_projective_transform(center, -angles, scale)
    transform[..., 3] += translations  # tx/ty/tz

    # pad transform to get Bx3x3
    transform_h = convert_affine_matrix_to_homography3d(transform)
    if any(s is not None for s in [sxy, sxz, syx, syz, szx, szy]):
        shear_mat   = get_shear_matrix3d(center, sxy, sxz, syx, syz, szx, szy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_rotation_matrix2d(
    center: Tensor, angle: Tensor, scale: Tensor
) -> Tensor:
    r"""Calculate an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})

    The transformation maps the rotation center to itself. If this is not the
    target, adjust the shift.

    Args:
        center (Tensor):
            Center of the rotation in the source image with shape [B, 2].
        angle (Tensor):
            Rotation angle in degrees. Positive values mean counter-clockwise
            rotation (the coordinate origin is assumed to be the top-left
            corner) with shape [B].
        scale (Tensor):
            Scale factor for x, y scaling with shape [B, 2].

    Returns:
        (Tensor):
            Affine matrix of 2D rotation with shape [B, 2, 3].

    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones((1, 2))
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        image([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_affine`.
    """
    if not isinstance(center, Tensor):
        raise TypeError(f"Input center type is not a Tensor. Got: {type(center)}.")

    if not isinstance(angle, Tensor):
        raise TypeError(f"Input angle type is not a Tensor. Got: {type(angle)}.")

    if not isinstance(scale, Tensor):
        raise TypeError(f"Input scale type is not a Tensor. Got: {type(scale)}.")

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError(f"Input center must be a Bx2 image. Got: {center.shape}.")

    if not len(angle.shape) == 1:
        raise ValueError(f"Input angle must be a B image. Got: {angle.shape}.")

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError(f"Input scale must be a [B, 2] image. Got: {scale.shape}.")

    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            f"Inputs must have same batch size dimension. "
            f"Got: center {center.shape}, angle {angle.shape}, "
            f"and scale {scale.shape}"
        )

    if not (center.device == angle.device == scale.device) \
        or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError(
            f"Inputs must have same device "
            f"Got: center ({center.device}, {center.dtype}), "
            f"angle ({angle.device}, {angle.dtype}), "
            f"and scale ({scale.device}, {scale.dtype})"
        )

    shift_m           = eye_like(3, center)
    shift_m[:, :2, 2] = center

    shift_m_inv           = eye_like(3, center)
    shift_m_inv[:, :2, 2] = -center

    scale_m           = eye_like(3, center)
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m            = eye_like(3, center)
    rotat_m[:, :2, :2] = angle_to_rotation_matrix(angle)

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3


def get_shear_matrix2d(
    center: Tensor, sx: Optional[Tensor] = None, sy: Optional[Tensor] = None
):
    r"""Compose shear matrix Bx4x4 from the components.

    Note: Ordered shearing, shear x-axis then y-axis.

    .. math::
        \begin{bmatrix}
            1 & b \\
            a & ab + 1 \\
        \end{bmatrix}

    Args:
        center (Tensor):
            Shearing center coordinates of (x, y).
        sx (Tensor, optional):
            Shearing degree along x axis.
        sy (Tensor, optional):
            Shearing degree along y axis.

    Returns:
        params to be passed to the affine transformation with shape [B, 3, 3].

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> sx = torch.randn(1)
        >>> sx
        image([1.5410])
        >>> center = torch.image([[0., 0.]])  # Bx2
        >>> get_shear_matrix2d(center, sx=sx)
        image([[[  1.0000, -33.5468,   0.0000],
                 [ -0.0000,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   1.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_affine`,
        :func:`warp_perspective`.
    """
    sx = torch.tensor([0.0]).repeat(center.size(0)) if sx is None else sx
    sy = torch.tensor([0.0]).repeat(center.size(0)) if sy is None else sy

    x, y = torch.split(center, 1, dim=-1)
    x, y = x.view(-1), y.view(-1)

    sx_tan    = torch.tan(sx)  # type: ignore
    sy_tan    = torch.tan(sy)  # type: ignore
    ones      = torch.ones_like(sx)  # type: ignore
    shear_mat = torch.stack(
        [
            ones,
            -sx_tan,
            sx_tan * y,  # type: ignore
            -sy_tan,
            ones + sx_tan * sy_tan,
            sy_tan * (sx_tan * y + x),
        ],
        dim=-1,
    ).view(-1, 2, 3)

    shear_mat = convert_affine_matrix_to_homography(shear_mat)
    return shear_mat


def _compute_shear_matrix_3d(
    sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan
):
    ones          = torch.ones_like(sxy_tan)  # type: ignore
    m00, m10, m20 = ones, sxy_tan, sxz_tan
    m01, m11, m21 = syx_tan, sxy_tan * syx_tan + ones, sxz_tan * syx_tan + syz_tan
    m02           = syx_tan * szy_tan + szx_tan
    m12           = sxy_tan * szx_tan + szy_tan * m11
    m22           = sxz_tan * szx_tan + szy_tan * m21 + ones
    return m00, m10, m20, m01, m11, m21, m02, m12, m22


def get_shear_matrix3d(
    center: Tensor,
    sxy   : Optional[Tensor] = None,
    sxz   : Optional[Tensor] = None,
    syx   : Optional[Tensor] = None,
    syz   : Optional[Tensor] = None,
    szx   : Optional[Tensor] = None,
    szy   : Optional[Tensor] = None,
):
    r"""Compose shear matrix Bx4x4 from the components.
    Note: Ordered shearing, shear x-axis then y-axis then z-axis.

    .. math::
        \begin{bmatrix}
            1 & o & r & oy + rz \\
            m & p & s & mx + py + sz -y \\
            n & q & t & nx + qy + tz -z \\
            0 & 0 & 0 & 1  \\
        \end{bmatrix}
        Where:
        m = S_{xy}
        n = S_{xz}
        o = S_{yx}
        p = S_{xy}S_{yx} + 1
        q = S_{xz}S_{yx} + S_{yz}
        r = S_{zx} + S_{yx}S_{zy}
        s = S_{xy}S_{zx} + (S_{xy}S_{yx} + 1)S_{zy}
        t = S_{xz}S_{zx} + (S_{xz}S_{yx} + S_{yz})S_{zy} + 1

    Params:
        center (Tensor):
            Shearing center coordinates of [x, y, z].
        sxy (Tensor, optional):
            Shearing degree along x axis, towards y plane.
        sxz (Tensor, optional):
            Shearing degree along x axis, towards z plane.
        syx (Tensor, optional):
            Shearing degree along y axis, towards x plane.
        syz (Tensor, optional):
            Shearing degree along y axis, towards z plane.
        szx (Tensor, optional):
            Shearing degree along z axis, towards x plane.
        szy (Tensor, optional):
            Shearing degree along z axis, towards y plane.

    Returns:
        params to be passed to the affine transformation.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> sxy, sxz, syx, syz = torch.randn(4, 1)
        >>> sxy, sxz, syx, syz
        (image([1.5410]), image([-0.2934]), image([-2.1788]), image([0.5684]))
        >>> center = torch.image([[0., 0., 0.]])  # Bx3
        >>> get_shear_matrix3d(center, sxy=sxy, sxz=sxz, syx=syx, syz=syz)
        image([[[  1.0000,  -1.4369,   0.0000,   0.0000],
                 [-33.5468,  49.2039,   0.0000,   0.0000],
                 [  0.3022,  -1.0729,   1.0000,   0.0000],
                 [  0.0000,   0.0000,   0.0000,   1.0000]]])

    .. note::
        This function is often used in conjunction with :func:`warp_perspective3d`.
    """
    sxy = torch.tensor([0.0]).repeat(center.size(0)) if sxy is None else sxy
    sxz = torch.tensor([0.0]).repeat(center.size(0)) if sxz is None else sxz
    syx = torch.tensor([0.0]).repeat(center.size(0)) if syx is None else syx
    syz = torch.tensor([0.0]).repeat(center.size(0)) if syz is None else syz
    szx = torch.tensor([0.0]).repeat(center.size(0)) if szx is None else szx
    szy = torch.tensor([0.0]).repeat(center.size(0)) if szy is None else szy

    x, y, z = torch.split(center, 1, dim=-1)
    x, y, z = x.view(-1), y.view(-1), z.view(-1)
    # Prepare parameters
    sxy_tan = torch.tan(sxy)  # type: ignore
    sxz_tan = torch.tan(sxz)  # type: ignore
    syx_tan = torch.tan(syx)  # type: ignore
    syz_tan = torch.tan(syz)  # type: ignore
    szx_tan = torch.tan(szx)  # type: ignore
    szy_tan = torch.tan(szy)  # type: ignore

    # compute translation matrix
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan
    )

    m03 = m01 * y + m02 * z
    m13 = m10 * x + m11 * y + m12 * z - y
    m23 = m20 * x + m21 * y + m22 * z - z

    # shear matrix is implemented with negative values
    sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan = -sxy_tan, -sxz_tan, -syx_tan, -syz_tan, -szx_tan, -szy_tan
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan
    )

    shear_mat = torch.stack([m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23], dim=-1).view(-1, 3, 4)
    shear_mat = convert_affine_matrix_to_homography3d(shear_mat)

    return shear_mat


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
