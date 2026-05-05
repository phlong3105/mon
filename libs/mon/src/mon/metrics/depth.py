#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth Metrics.

This module provides functions and classes to evaluate the quality of depth maps.
"""

from __future__ import annotations

__all__ = [
    "compute_depth_metrics",
]

import numpy as np
from numpy import ndarray


# ==============================================================================
# region Depth Metrics
# ==============================================================================

def compute_depth_metrics(
    pred: ndarray,
    target: ndarray,
    valid_mask: ndarray | None = None,
    normalize: bool = True,
) -> dict[str, float]:
    """Compute depth metrics between two depth maps.

    Args:
        pred (ndarray): Predicted depth map.
        target (ndarray): Target depth map.
        valid_mask (ndarray | None, optional): Optional boolean mask indicating
            valid pixels. If None, all pixels are considered valid. Defaults to None.
        normalize (bool, optional): Whether to normalize the predicted depth
            map to the same range as the target depth map before computing metrics.
            Defaults to True.

    Returns:
        dict[str, float]: A dictionary containing the computed depth metrics.
    """
    # Validate inputs
    if pred.shape != target.shape:
        raise ValueError(
            f"expected pred and target to have the same shape, "
            f"got {pred.shape} != {target.shape}."
        )

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    # Create valid_mask if not provided
    if valid_mask is None:
        valid_mask = (target > 0) & (~np.isnan(pred)) & (~np.isnan(target))
    if not np.any(valid_mask):
        raise RuntimeError(f"no valid pixels found in the depth maps.")

    # Normalize predicted depth map to target's range
    if normalize:
        valid_target = target[valid_mask]
        target_min, target_max = np.min(valid_target), np.max(valid_target)

        # Avoid division by zero in normalization
        if target_max == target_min:
            raise ValueError(
                f"target depth map has zero range (min == max), cannot normalize."
            )

        # Min-max normalization of predicted depths
        valid_pred = pred[valid_mask]
        pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)
        if pred_max != pred_min:  # Only normalize if the predicted map has a range
            pred = target_min + (target_max - target_min) * (pred - pred_min) / (pred_max - pred_min)
        else:
            # If the predicted map is constant, scale to target's mean or min
            pred = np.full_like(pred, target_min)

        # Update min/max of the normalized predicted map
        valid_pred = pred[valid_mask]
        pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)

    # Flatten arrays and apply mask
    pred_flat = pred[valid_mask]
    target_flat = target[valid_mask]

    # Compute differences
    diff = pred_flat - target_flat
    abs_diff = np.abs(diff)
    # Absolute Relative Error
    abs_rel = np.mean(abs_diff / target_flat)
    # Squared Relative Error
    sq_rel = np.mean((diff ** 2) / target_flat)
    # RMSE
    rmse = np.sqrt(np.mean(diff ** 2))
    # RMSE log
    log_pred = np.log(np.clip(pred_flat, 1e-10, None))  # Avoid log(0)
    log_target = np.log(np.clip(target_flat, 1e-10, None))
    rmse_log = np.sqrt(np.mean((log_pred - log_target) ** 2))
    # MAE
    mae = np.mean(abs_diff)
    # Threshold accuracies
    thresh = np.maximum(pred_flat / target_flat, target_flat / pred_flat)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "mae": mae,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
