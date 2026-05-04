#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO utilities.

This module provides various utilities for SALEO.
"""

from __future__ import annotations

__all__ = [
    "JitteredGridSampler",
    "RandomPixelSampler",
    "RgbToHsv",
    "RgbToHvi",
    "filter_up",
    "get_local_features",
    "get_nearest_features",
    "get_v_component",
    "interpolate_image",
    "replace_v_component",
]

import random

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from mon.core import Size
from mon.ops import FastGuidedFilter


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Color ---

class RgbToHsv(nn.Module):
    """A convenience class to convert RGB images to HSV color space and back."""

    # --- Callable & Context Manager ---
    def from_rgb(self, rgb: Tensor) -> Tensor:
        """Convert an RGB image to HSV color space.

        Args:
            rgb (Tensor): An RGB image tensor of shape (B, 3, H, W) and pixel
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: An HSV image tensor of shape (B, 3, H, W) and pixel values
                ranging from 0.0 to 1.0.
        """
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.0
        hsv_h /= 6.0
        hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def to_rgb(self, hsv: Tensor) -> Tensor:
        """Convert an HSV image to RGB color space.

        Args:
            hsv (Tensor): An HSV image tensor of shape (B, 3, H, W) and pixel
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: An RGB image tensor of shape (B, 3, H, W) and pixel values
                ranging from 0.0 to 1.0.
        """
        hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        _c = hsv_l * hsv_s
        _x = _c * (- torch.abs(hsv_h * 6. % 2.0 - 1) + 1.)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c)
        idx = (hsv_h * 6.0).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsv)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb


class RgbToHvi(nn.Module):
    """A module for converting RGB images to HVI color space and back.

    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-8, requires_grad: bool = False):
        """Initialize a new instance.

        Args:
            eps (float): Epsilon value to avoid division by zero. Defaults to 1e-8.
            requires_grad (bool): If True, allows gradient computation for
                ``density_k``. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.pi = 3.141592653589793
        self.eps = eps
        # Learnable 'k' controls the color gamut sensitivity relative to intensity
        self.density_k = nn.Parameter(
            torch.full([1], 0.1),
            requires_grad=requires_grad
        )

    # --- Callable & Context Manager ---
    def from_rgb(self, rgb: Tensor) -> Tensor:
        """Convert an RGB image to HVI color space.

        Args:
            rgb (Tensor): An RGB image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: The corresponding HVI image tensor of shape (B, 3, H, W)
                with H and V values ranging from -1.0 to 1.0, and I values
                ranging from 0.0 to 1.0.
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        max_val, _ = rgb.max(1)
        min_val, _ = rgb.min(1)
        diff = max_val - min_val + self.eps

        # Standard Hue calculation (0-6 range)
        hue = torch.zeros_like(max_val)
        hue = torch.where(max_val == r, (g - b) / diff % 6, hue)
        hue = torch.where(max_val == g, (b - r) / diff + 2, hue)
        hue = torch.where(max_val == b, (r - g) / diff + 4, hue)
        hue = torch.where(diff < self.eps, torch.zeros_like(hue), hue)
        hue = hue / 6.0  # Normalized to [0, 1]

        # Saturation and Intensity
        # S = (max - min) / max
        s = diff / (max_val + self.eps)
        i = max_val # Intensity (Value)

        # Cartesian Mapping (The HVI specific logic)
        # Sensitivity curves the color response based on Intensity
        sensitivity = ((i * 0.5 * self.pi).sin() + self.eps).pow(self.density_k)

        # Mapping Hue/Saturation (Polar) -> H/V (Cartesian)
        h_coord = sensitivity * s * torch.cos(2.0 * self.pi * hue)
        v_coord = sensitivity * s * torch.sin(2.0 * self.pi * hue)

        return torch.stack([h_coord, v_coord, i], dim=1)

    def to_rgb(self, hvi: Tensor) -> Tensor:
        """Convert an HVI image to RGB color space.

        Args:
            hvi (Tensor): A HVI image tensor of shape (B, 3, H, W) with H and V
                values ranging from -1.0 to 1.0, and I values ranging from
                0.0 to 1.0.

        Returns:
            Tensor: The corresponding RGB image tensor of shape (B, 3, H, W)
                and values ranging from 0.0 to 1.0.
        """
        h_coord, v_coord, i = hvi[:, 0, :, :], hvi[:, 1, :, :], hvi[:, 2, :, :]

        sensitivity = ((i * 0.5 * self.pi).sin() + self.eps).pow(self.density_k)

        # Reconstruct Hue (angle) and Saturation (magnitude)
        h = torch.atan2(v_coord, h_coord) / (2.0 * self.pi) % 1.0
        s = torch.sqrt(h_coord ** 2 + v_coord ** 2 + self.eps) / (sensitivity + self.eps)
        s = torch.clamp(s, 0, 1)

        # HSV to RGB Conversion (Simplified Vectorized)
        hi = (h * 6.0).floor()
        f = h * 6.0 - hi
        p = i * (1.0 - s)
        q = i * (1.0 - (f * s))
        t = i * (1.0 - ((1.0 - f) * s))

        # Reconstruct RGB channels based on Hue sector
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        # Hi indices: 0: R,Gt,Bp | 1: Rq,G,Bp | 2: Rp,G,Bt | 3: Rp,Gq,B | 4: Rt,Gp,B | 5: R,Gp,Bq
        mask = (hi == 0); r[mask], g[mask], b[mask] = i[mask], t[mask], p[mask]
        mask = (hi == 1); r[mask], g[mask], b[mask] = q[mask], i[mask], p[mask]
        mask = (hi == 2); r[mask], g[mask], b[mask] = p[mask], i[mask], t[mask]
        mask = (hi == 3); r[mask], g[mask], b[mask] = p[mask], q[mask], i[mask]
        mask = (hi == 4); r[mask], g[mask], b[mask] = t[mask], p[mask], i[mask]
        mask = (hi == 5); r[mask], g[mask], b[mask] = i[mask], p[mask], q[mask]

        return torch.stack([r, g, b], dim=1)


def get_v_component(image_hsv: Tensor) -> Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return image_hsv[:, -1].unsqueeze(0)


def replace_v_component(image_hsv: Tensor, v_new: Tensor) -> Tensor:
    """Replaces the V component of an HSV image (1, 3, H, W)."""
    image_hsv[:, -1] = v_new
    return image_hsv


# --- Features ---

def get_local_features(image: Tensor, kernel_size: int = 7) -> Tensor:
    """Extract local neighborhoods (patches) for every pixel in the image.

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and pixel values
            ranging from 0.0 to 1.0.
        kernel_size (int): Size of the local window. Defaults to 7.

    Returns:
        Tensor: Sample features tensor of shape (B, H, W, C*K*K). Every spatial
            position (H, W) contains the flattened vector of its neighbors.
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to be Tensor of shape (B, C, H, W), "
            f"but got {image.shape}."
        )

    b, c, h, w = image.shape
    k = kernel_size
    padding = k // 2

    # 1. Extracts all sliding windows.
    # Output shape: (b, c*k*k, h*w)
    unfolded = F.unfold(image, kernel_size=k, padding=padding)

    # 2. Reshape to restore spatial dimensions
    # Output shape: (b, c*k*k, h, w)
    unfolded = unfolded.view(b, c * k * k, h, w)

    # 3. Permute to put features last (standard for Linear layers)
    # Output shape: (b, h, w, c*k*k)
    features = unfolded.permute(0, 2, 3, 1).contiguous()

    return features


def get_nearest_features(
    image: Tensor,
    query_coords: Tensor,
    kernel_size: int = 7,
) -> Tensor:
    """Query arbitrary-scale features.

    Extract the nearest neighbor features in input ``image`` for ``query_coords``.

    Args:
        image (Tensor): Image tensor of shape (B, C, H_in, W_in) and pixel
            values ranging from 0.0 to 1.0.
        query_coords (Tensor): Query coordinates tensor of shape
            (B, H_out, W_out, 2) and values ranging from -1.0 to 1.0.
        kernel_size (int): Size of the local window. Defaults to 7.

    Returns:
        Tensor: Sample features tensor of shape (B, H_out, W_out, C*K*K). Every
            spatial position (H_out, W_out) contains the flattened vector of its
            neighbors in the input ``image``.
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to be Tensor of shape (B, C, H, W), "
            f"but got {image.shape}."
        )

    b, c, h_in, w_in = image.shape
    k = kernel_size
    padding = k // 2

    # 1. Create the dense feature map
    # Output shape: (b, c*k*k, h_in*w_in)
    unfolded = F.unfold(image, kernel_size=k, padding=padding)

    # 2. Reshape to restore spatial dimensions
    # Output shape: (b, c*k*k, h_in, w_in)
    feature_map = unfolded.view(b, c * k * k, h_in, w_in)

    # 3. Query the feature map
    # Output shape: (b, h_out, w_out, c*k*k)
    sampled_map = F.grid_sample(
        feature_map,
        query_coords,
        mode="nearest",
        padding_mode="border",
        align_corners=False
    )

    # 4. Permute to put features last (standard for Linear layers)
    # Output shape: (b, h_out, w_out, c*k*k)
    sampled_features = sampled_map.permute(0, 2, 3, 1).contiguous()

    return sampled_features


# --- Resize ---

def interpolate_image(image: Tensor, size: Size) -> Tensor:
    """Reshapes the image based on new resolution."""
    size = Size.from_value(size)
    return F.interpolate(image, size=size.hw)


def filter_up(x_lr: Tensor, y_lr: Tensor, x_hr: Tensor, r: int = 1):
    """Applies the guided filter to upscale the predicted image."""
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr


# --- Samplers ---

class JitteredGridSampler:
    """A sampler that synchronously splits input maps (e.g., image, depth, etc.)
    into batches of grid patches (ViT style).

    Ensure full coverage of the input maps, but applies a random spatial offset
    (jitter) every epoch.

    Advantages:
        1. Guarantees the network sees EVERY pixel (fast convergence).
        2. Random offset prevents 'grid artifacts' or seams in the output.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        patch_size: int = 64,
        device: torch.device | None = None,
    ):
        """Initialize a new instance.

        Args:
            image (Tensor): Image tensor of shape (1, C, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (1, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None means no
                accompanying for this ``image``.
            patch_size (int, optional): Size of square patches. Defaults to 64.
            device (torch.device, optional): Device to use for computation.
                By default, uses ``image``'s device.
        """
        # Assign attributes
        self.image = image
        self.depth = depth
        self.patch_size = patch_size
        self.imgsz = Size.from_value(image)
        self.device = device or image.device

        # Move the device
        if self.image.device != self.device:
            self.image = self.image.to(self.device)
        if self.depth is not None and self.depth.device != self.device:
            self.depth = self.depth.to(self.device)

    # --- Properties ---
    @property
    def has_depth(self) -> bool:
        """Return True if a depth map is provided."""
        return self.depth is not None

    # --- Callable & Context Manager ---
    def get_epoch_iterator(self, batch_size: int = 1):
        """Yield batches of grid patches from the inputs.

        Call this at the start of every epoch for one full sweep over the input
        maps (e.g., image, depth, etc.).

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            A tuple of patches tensor of shape (batch_size, C, patch_size, patch_size).
            In addition, the corresponding coordinates tensor shape
            (batch_size, patch_size, patch_size, 2) and values ranging from
            -1.0 to 1.0.

        Examples:
            >>> # Get the iterator for THIS epoch (new jitter every time)
            >>> sampler  = JitteredGridSampler(...)
            >>> iterator = sampler.get_epoch_iterator(batch_size=4)
            >>> for batch_i, batch_d, coords in iterator:
            >>>     pass
        """
        H, W = self.imgsz
        P = self.patch_size

        # 1. Generate random jitter for this entire epoch
        # We shift the grid origin by random (dx, dy)
        dx = torch.randint(0, P, (1,)).item()
        dy = torch.randint(0, P, (1,)).item()

        # 2. Create grid points (top-left corners)
        # We start slightly off-image (negative index) to ensure the jitter
        # doesn't leave gaps at the top/left borders.
        h_starts = range(dy - P, H, P)
        w_starts = range(dx - P, W, P)
        patches_list = []

        for h in h_starts:
            for w in w_starts:
                # Clamp coordinates to stay within image bounds
                # This handles the "edges" of the image naturally
                h_real = max(0, min(h, H - P))
                w_real = max(0, min(w, W - P))
                patches_list.append((h_real, w_real))

        # 3. Shuffle the grid order
        # Crucial! We don't want the network to learn a "sweep" pattern.
        random.shuffle(patches_list)

        # 4. Batch accumulation
        batch_image = []
        batch_depth = []
        batch_coords = []

        for (h, w) in patches_list:
            # 4.1. Crop input maps
            image_crop = self.image[:, :, h:h+P, w:w+P]
            depth_crop = self.depth[:, :, h:h+P, w:w+P] if self.has_depth else None

            # 4.2. Generate continuous coordinates
            # Map pixels (h..h+P) to global normalized coordinates (-1..1)
            y_indices = torch.arange(h, h + P)
            x_indices = torch.arange(w, w + P)
            y_mesh, x_mesh = torch.meshgrid(y_indices, x_indices, indexing="ij")

            # Normalize to [-1, 1]
            x_norm = (x_mesh / (W - 1)) * 2 - 1
            y_norm = (y_mesh / (H - 1)) * 2 - 1

            # Stack to get (P, P, 2) and expand to (1, P, P, 2)
            coord_crop = torch.stack([x_norm, y_norm], dim=-1)  # (P, P, 2)
            coord_crop = coord_crop.unsqueeze(0)  # (1, P, P, 2)

            # 4.3. Append to list
            batch_image.append(image_crop)
            batch_depth.append(depth_crop)
            batch_coords.append(coord_crop)

            # Yield when the batch is full
            if len(batch_image) == batch_size:
                yield (
                    torch.cat(batch_image),
                    torch.cat(batch_depth) if self.has_depth else None,
                    torch.cat(batch_coords)
                )
                # Reset buffers
                batch_image = []
                batch_depth = []
                batch_coords = []

        # Yield any remaining patches (last incomplete batch)
        if len(batch_image) > 0:
            yield (
                torch.cat(batch_image),
                torch.cat(batch_depth) if self.has_depth else None,
                torch.cat(batch_coords)
            )


class RandomPixelSampler:
    """A sampler that randomly samples massive batches of random pixels and
    their neighbors
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        window_size: int = 3,
        device: torch.device | None = None,
    ):
        """Initialize a new instance.

        Args:
            image (Tensor): Image tensor of shape (1, C, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (1, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None means no
                accompanying for this ``image``.
            window_size (int, optional): Size of the local window to extract
                around each sampled pixel. Defaults to 3 (i.e., 3x3 neighborhood).
            device (torch.device, optional): Device to use for computation.
                By default, uses ``image``'s device.
        """
        # Assign attributes
        self.image = image
        self.depth = depth
        self.window_size = window_size
        self.imgsz = Size.from_value(image)
        self.device = device or image.device

        # We blur the input for feature extraction to prevent the network
        # from learning high-frequency noise from the neighbors.
        self.image_blur = TF.gaussian_blur(
            self.image,
            kernel_size=[5, 5],
            sigma=[1.5, 1.5]
        )

    # --- Properties ---
    @property
    def has_depth(self) -> bool:
        """Return True if a depth map is provided."""
        return self.depth is not None

    # --- Callable & Context Manager ---
    def sample_batch(
        self,
        batch_size: int = 500000
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a random batch of coordinates and their features
        (Instant sampling. Zero overhead.)

        Args:
            batch_size (int, optional): Number of random pixels to sample.
                Defaults to 500,000.

        Returns:
            A tuple containing:
                - Sampled coordinates tensor of shape (batch_size, 2) and
                  values ranging from -1.0 to 1.0.
                - Sampled image features tensor of shape (batch_size, ...).
                - Sampled depth features tensor of shape (batch_size, ...)
                  if a depth map is provided.
        """
        # 1. Generate Random Coordinates [-1, 1]
        r_x = torch.rand(batch_size, device=self.device) * 2 - 1
        r_y = torch.rand(batch_size, device=self.device) * 2 - 1

        # Shape: (1, 1, N, 2) for grid_sample
        # We need requires_grad=True for the smoothness loss later
        coords = torch.stack([r_x, r_y], dim=-1).view(1, 1, -1, 2).requires_grad_(True)

        # 2. Extract features
        # features_i = get_nearest_features(self.image, coords, self.window_size)
        features_i = get_nearest_features(self.image_blur, coords, self.window_size)

        if self.has_depth:
            features_d = get_nearest_features(self.depth, coords, self.window_size)
            features = torch.cat([features_i, features_d], dim=-1)  # (B, 64, 64, 18)
        else:
            features_d = None
            features = features_i

        # 3. Extract targets (from SHARP/Original image)
        target = F.grid_sample(self.image, coords, mode="nearest", align_corners=False)

        # 4. Flatten for MLP
        coords = coords.to(self.device)
        # features_i = features_i.to(self.device)
        # features_d = features_d.to(self.device) if features_d is not None else None
        target = target.to(self.device)

        return coords, features, target

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
