#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO utilities.

This module provides various utilities for SALEO.
"""

from __future__ import annotations

__all__ = [
    "JitteredGridSampler",
    "RandomPixelSampler",
    "get_local_features",
    "get_nearest_features",
    "get_v_component",
    "hsv_to_rgb",
    "replace_v_component",
    "rgb_to_hsv",
]

import random

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from mon import parse_imgsz


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Color Utils ---

def rgb_to_hsv(rgb: Tensor) -> Tensor:
    """Convert an RGB image to HSV color space.

    Args:
        rgb (Tensor): An RGB image tensor of shape (B, 3, H, W) and pixel values
            ranging from 0.0 to 1.0.

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


def hsv_to_rgb(hsv: Tensor) -> Tensor:
    """Convert an HSV image to RGB color space.

    Args:
        hsv (Tensor): An HSV image tensor of shape (B, 3, H, W) and pixel values
            ranging from 0.0 to 1.0.

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


def get_v_component(img_hsv: Tensor) -> Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return img_hsv[:, -1].unsqueeze(0)


def replace_v_component(img_hsv: Tensor, v_new: Tensor) -> Tensor:
    """Replaces the V component of a HSV image (1, 3, H, W)."""
    img_hsv[:, -1] = v_new
    return img_hsv


# --- Features Utils ---

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
        self.imgsz = parse_imgsz(image)
        self.device = device or image.device

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
            A tuple of patches, each formatted as a Tensor of shape
            (batch_size, C, patch_size, patch_size). In addition, the
            corresponding coordinates, formatted as a Tensor shape
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
        self.imgsz = parse_imgsz(image)
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
                - Sampled coordinates, formatted as a Tensor of shape
                  (batch_size, 2) and values ranging from -1.0 to 1.0.
                - Sampled image features, formatted as a Tensor of shape
                  (batch_size, ...).
                - Sampled depth features, formatted as a Tensor of shape
                  (batch_size, ...) if a depth map is provided.
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

        # 3. Extract Targets (from SHARP/Original image)
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
