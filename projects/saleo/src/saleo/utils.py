#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO utilities.

This module provides various utilities for SALEO.
"""

from __future__ import annotations

__all__ = [
    "ContinuousPatchSampler",
    "JitteredGridSampler",
    "get_local_features",
    "get_nearest_features",
]

import random
from typing import Optional

import torch
import torch.nn.functional as F

from mon.core import image as I


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Features Query ---

def get_local_features(image: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Extract local neighborhoods (patches) for every pixel in the image.

    Args:
        image: Image, formatted as a torch.Tensor of shape (B, C, H, W) and
            values ranging from 0.0 to 1.0.
        kernel_size: Size of the local window. Defaults to 7.

    Returns:
        Sample features, formatted as a torch.Tensor of shape (B, H, W, C*K*K).
        Every spatial position (H, W) contains the flattened vector of its
        neighbors.
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to be torch.Tensor of shape (B, C, H, W), "
            f"but got {image.shape}"
        )

    b, c, h, w = image.shape
    k          = kernel_size
    padding    = k // 2

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
    image       : torch.Tensor,
    query_coords: torch.Tensor,
    kernel_size : int = 7
) -> torch.Tensor:
    """Query arbitrary-scale features.

    Extract the nearest neighbor features in input ``image`` for ``query_coords``.

    Args:
        image: Image, formatted as a torch.Tensor of shape (B, C, H_in, W_in)
            and values ranging from 0.0 to 1.0.
        query_coords: Target coordinates, formatted as a torch.Tensor of shape
            (B, H_out, W_out, 2) and values ranging from -1.0 to 1.0.
        kernel_size: Size of the local window. Defaults to 7.

    Returns:
        Sample features, formatted as a torch.Tensor of shape (B, H_out, W_out, C*k*K).
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to be torch.Tensor of shape (B, C, H, W), "
            f"but got {image.shape}"
        )

    b, c, h_in, w_in = image.shape
    k       = kernel_size
    padding = k // 2

    # 1. Create the dense feature map
    # Output shape: (b, c*k*k, h_in*w_in)
    unfolded    = F.unfold(image, kernel_size=k, padding=padding)

    # 2. Reshape to restore spatial dimensions
    # Output shape: (b, c*k*k, h_in, w_in)
    feature_map = unfolded.view(b, c * k * k, h_in, w_in)

    # 3. Query the feature map
    # Output shape: (b, h_out, w_out, c*k*k)
    sampled_map = F.grid_sample(
        feature_map,
        query_coords,
        mode          = "nearest",
        padding_mode  = "border",
        align_corners = False
    )

    # 4. Permute to put features last (standard for Linear layers)
    # Output shape: (b, h_out, w_out, c*k*k)
    sampled_features = sampled_map.permute(0, 2, 3, 1).contiguous()

    return sampled_features


# --- Patch Samplers ---

class ContinuousPatchSampler:
    """A sampler that randomly samples patches from input maps (e.g., image,
    depth, etc.) at each training iteration/step.

    Attributes:
        image (torch.Tensor): Image, formatted as a torch.Tensor of shape
            (1, C, H, W) and values ranging from 0.0 to 1.0.
        depth (torch.Tensor): Optional depth map, formatted as a torch.Tensor
            of shape (1, 1, H, W) and values ranging from 0.0 to 1.0.
        patch_size (int): Size of square patches.
        imgsz (tuple[int, int]): Original image size (H, W).
        device (torch.device): Device to use for computation. By default, uses
            ``image``'s device.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        image     : torch.Tensor,
        depth     : torch.Tensor,
        patch_size: int = 64,
    ):
        """Initialize a new instance.

        Args:
            image: image, formatted as a torch.Tensor of shape (1, C, H, W)
                and values ranging from 0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape (1, 1, H, W)
                and values ranging from 0.0 to 1.0.
            patch_size: Size of square patches. Defaults to 64.
        """
        # Assign attributes
        self.image      = image
        self.depth      = depth
        self.patch_size = patch_size
        self.imgsz      = I.imgsz(image)
        self.device     = image.device

        if self.depth is not None and self.depth.device != self.device:
            self.depth = self.depth.to(self.device)

    # --- Properties ---
    @property
    def has_depth(self) -> bool:
        """Return True if a depth map is provided."""
        return self.depth is not None

    # --- Callable & Context Manager ---
    def get_random_patches(
        self, k: int = 1
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Sample ``k`` random patches from the inputs.

        Args:
            k: Number of patches to sample.

        Returns:
            A tuple of patches, each formatted as a torch.Tensor of shape
            (batch_size, C, patch_size, patch_size). In addition, the
            corresponding coordinates, formatted as a torch.Tensor shape
            (batch_size, patch_size, patch_size, 2) and values ranging from
            -1.0 to 1.0.
        """
        H, W = self.imgsz
        P    = self.patch_size

        patches_image  = []
        patches_depth  = []
        patches_coords = []

        for _ in range(k):
            # 1. Select random top-left corner
            h_start = torch.randint(0, H - P, (1,)).item()
            w_start = torch.randint(0, W - P, (1,)).item()
            h_end   = h_start + P
            w_end   = w_start + P

            # 2. Crop input maps (image, depth)
            image_crop = self.image[:, :, h_start:h_end, w_start:w_end]
            depth_crop = self.depth[:, :, h_start:h_end, w_start:w_end] if self.has_depth else None

            # 3. Generate coordinates for THIS specific patch
            # We map the global pixel indices to global -1..1 space
            # Formula: -1 + (2 * pixel_idx / total_dim)

            # Generate local grid indices
            y_indices      = torch.arange(h_start, h_end)
            x_indices      = torch.arange(w_start, w_end)
            y_mesh, x_mesh = torch.meshgrid(y_indices, x_indices, indexing="ij")

            # Normalize to [-1, 1]
            x_norm = (x_mesh / (W - 1)) * 2 - 1
            y_norm = (y_mesh / (H - 1)) * 2 - 1

            # Stack to get (P, P, 2) and expand to (1, P, P, 2)
            coord_crop = torch.stack([x_norm, y_norm], dim=-1)  # (P, P, 2)
            coord_crop = coord_crop.unsqueeze(0)  # (1, P, P, 2)

            # 4. Append to list
            patches_image.append(image_crop)
            patches_depth.append(depth_crop)
            patches_coords.append(coord_crop)

        # Concatenate and return
        return (
            torch.cat(patches_image,  dim=0),
            torch.cat(patches_depth,  dim=0) if self.has_depth else None,
            torch.cat(patches_coords, dim=0)
        )


class JitteredGridSampler:
    """A sampler that synchronously splits input maps (e.g., image, depth, etc.)
    into batches of grid patches (ViT style).

    Ensure full coverage of the input maps, but applies a random spatial offset
    (jitter) every epoch.

    Advantage:
        1. Guarantees the network sees EVERY pixel (fast convergence).
        2. Random offset prevents 'grid artifacts' or seams in the output.

    Attributes:
        image (torch.Tensor): Image, formatted as a torch.Tensor of shape
            (1, C, H, W) and values ranging from 0.0 to 1.0.
        depth (torch.Tensor): Optional depth map, formatted as a torch.Tensor
            of shape (1, 1, H, W) and values ranging from 0.0 to 1.0.
        patch_size (int): Size of square patches.
        imgsz (tuple[int, int]): Original image size (H, W).
        device (torch.device): Device to use for computation. By default, uses
            ``image``'s device.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        image     : torch.Tensor,
        depth     : torch.Tensor = None,
        patch_size: int          = 64,
    ):
        """Initialize a new instance.

        Args:
            image: Image, formatted as a torch.Tensor of shape (1, C, H, W)
                and values ranging from 0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape (1, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None means no
                accompanying depth map is provided for this ``image``.
            patch_size: Size of square patches. Defaults to 64.
        """
        # Assign attributes
        self.image      = image
        self.depth      = depth
        self.patch_size = patch_size
        self.imgsz      = I.imgsz(image)
        self.device     = image.device

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
            batch_size: Batch size. Defaults to 1.

        Returns:
            A tuple of patches, each formatted as a torch.Tensor of shape
            (batch_size, C, patch_size, patch_size). In addition, the
            corresponding coordinates, formatted as a torch.Tensor shape
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
        P    = self.patch_size

        # 1. Generate random jitter for this entire epoch
        # We shift the grid origin by random (dx, dy)
        dx = torch.randint(0, P, (1,)).item()
        dy = torch.randint(0, P, (1,)).item()

        # 2. Create grid points (top-left corners)
        # We start slightly off-image (negative index) to ensure the jitter
        # doesn't leave gaps at the top/left borders.
        h_starts     = range(dy - P, H, P)
        w_starts     = range(dx - P, W, P)
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
        batch_image  = []
        batch_depth  = []
        batch_coords = []

        for (h, w) in patches_list:
            # 4.1. Crop input maps
            image_crop = self.image[:, :, h:h+P, w:w+P]
            depth_crop = self.depth[:, :, h:h+P, w:w+P] if self.has_depth else None

            # 4.2. Generate continuous coordinates
            # Map pixels (h..h+P) to global normalized coordinates (-1..1)
            y_indices      = torch.arange(h, h + P)
            x_indices      = torch.arange(w, w + P)
            y_mesh, x_mesh = torch.meshgrid(y_indices, x_indices, indexing="ij")

            # Stack to get (P, P, 2) and expand to (1, P, P, 2)
            x_norm     = (x_mesh / (W - 1)) * 2 - 1
            y_norm     = (y_mesh / (H - 1)) * 2 - 1
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
                batch_image  = []
                batch_depth  = []
                batch_coords = []

        # Yield any remaining patches (last incomplete batch)
        if len(batch_image) > 0:
            yield (
                torch.cat(batch_image),
                torch.cat(batch_depth) if self.has_depth else None,
                torch.cat(batch_coords)
            )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
