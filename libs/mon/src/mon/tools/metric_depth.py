#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth metric evaluators.

This module provides various depth metric evaluators.
"""

from __future__ import annotations

__all__ = [
    "DepthEvaluator",
]

import argparse
import logging

import albumentations as A
import box
import matplotlib
import numpy as np

from mon.core import (
    console,
    create_device,
    create_progress_bar,
    image as I,
    Path,
)
from mon.training.data import DataLoader, ImageEvalDataset

current_file = Path(__file__).normalize(exist=True)
current_dir  = current_file.parents[0]
_METRICS     = ["abs_rel", "sq_rel", "rmse", "rmse_log", "mae", "delta1", "delta2", "delta3"]


# ==============================================================================
# region CONTROL
# ==============================================================================

class DepthEvaluator:
    """A runner for measuring depth metrics."""

    # --- Lifecycle & Initialization ---
    def __init__(self, cfg: box.Box):
        # Assign attributes
        self._cfg       = cfg
        self.verbose    = cfg.verbose
        self._arch      = cfg.arch
        self._model     = cfg.model
        self._data      = cfg.data
        self._device    = create_device(cfg.device)
        self._imgsz     = I.imgsz(cfg.imgsz)
        self._resize    = cfg.resize
        self._normalize = cfg.normalize
        self._use_color = cfg.use_color
        self._save_txt  = cfg.save_txt

        # Resolve paths
        self._input_dir  = Path(cfg.input_dir).normalize(exist=True)
        self._target_dir = Path(cfg.target_dir).normalize(exist=True)

        # Setup components (lightweight)
        # Define metrics
        self._metrics = [m.lower() for m in _METRICS]
        self._cmap    = matplotlib.colormaps.get_cmap("Spectral_r")

        # Initialize states
        self._results = {}

    def _build_dataloader(self) -> DataLoader:
        """Create a dataloader for the given dataset."""
        if self._resize:
            h, w      = self._imgsz
            transform = A.Resize(height=h, width=w)
        else:
            transform = None

        return DataLoader(
            dataset = ImageEvalDataset(
                input_dir  = self._input_dir,
                target_dir = self._target_dir,
                transform  = transform,
                verbose    = self.verbose,
            ), batch_size = 1
        )

    # --- Callable & Context Manager ---
    def run(self):
        """Run the metric measurement process."""
        # Summarize the current run
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True
        console.rule(f"[bold red] {self._model}")
        console.log(f"[bold green]Model : {self._model}")
        console.log(f"[bold red]Data  : {self._data}")
        console.log(f"[bold]Device: {self._device}")

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        dataloader = self._build_dataloader()

        # Processing
        self._results = self._measure(dataloader=dataloader)

        # Print results
        self._print_results()

    def _measure(self, dataloader: DataLoader) -> dict:
        """Measure IQA metrics based on the configuration."""
        # Resolve attributes
        model     = self._model
        data      = self._data
        device    = self._device
        metrics   = self._metrics
        normalize = self._normalize
        cmap      = self._cmap
        use_color = self._use_color
        verbose   = self.verbose

        # Prepare containers
        values    = {m: [] for m in metrics}
        results   = {}

        # Processing loop
        with create_progress_bar(transient=not verbose) as pbar:
            desc = f"[bright_yellow]Measuring {model} | {data}"
            for i, datapoint in pbar.track(
                sequence    = enumerate(dataloader),
                total       = len(dataloader),
                description = desc
            ):
                image  = datapoint["image"]
                target = datapoint["target"]

                if use_color:
                    image  =  (cmap(image)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    target = (cmap(target)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                # Move tensors to a device
                image  =  image.to(device=device)
                target = target.to(device=device) if target is not None else None

                # Measure metric
                measured_results = self._compute_metrics(image, target)
                for k, v in measured_results.items():
                    if k in values:
                        values[k].append(v)

        # Aggregates results
        for m, v in values.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None

        return results

    def _compute_metrics(
        self,
        pred      : np.ndarray,
        target    : np.ndarray,
        valid_mask: np.ndarray = None,
    ) -> dict:
        # Input validation
        if pred.shape != target.shape:
            raise ValueError("Predicted and target depth maps must have the same shape.")

        pred   =   pred.astype(np.float32)
        target = target.astype(np.float32)

        # Create valid_mask if not provided
        if valid_mask is None:
            valid_mask = (target > 0) & (~np.isnan(pred)) & (~np.isnan(target))

        if not np.any(valid_mask):
            raise ValueError("No valid pixels found in the depth maps.")

        # Normalize predicted depth map to target's range
        if self._normalize:
            valid_target           = target[valid_mask]
            target_min, target_max = np.min(valid_target), np.max(valid_target)

            # Avoid division by zero in normalization
            if target_max == target_min:
                raise ValueError("Target depth map has no range (min equals max).")

            # Min-max normalization of predicted depths
            valid_pred         = pred[valid_mask]
            pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)
            if pred_max != pred_min:  # Only normalize if predicted map has a range
                pred = target_min + (target_max - target_min) * (pred - pred_min) / (pred_max - pred_min)
            else:
                # If predicted map is constant, scale to target's mean or min
                pred = np.full_like(pred, target_min)

            # Update min/max of normalized predicted map
            valid_pred         = pred[valid_mask]
            pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)

        # Flatten arrays and apply mask
        pred_flat   = pred[valid_mask]
        target_flat = target[valid_mask]

        # Compute differences
        diff       = pred_flat - target_flat
        abs_diff   = np.abs(diff)
        # Absolute Relative Error
        abs_rel    = np.mean(abs_diff / target_flat)
        # Squared Relative Error
        sq_rel     = np.mean((diff ** 2) / target_flat)
        # RMSE
        rmse       = np.sqrt(np.mean(diff ** 2))
        # RMSE log
        log_pred   = np.log(np.clip(pred_flat,   1e-10, None))  # Avoid log(0)
        log_target = np.log(np.clip(target_flat, 1e-10, None))
        rmse_log   = np.sqrt(np.mean((log_pred - log_target) ** 2))
        # MAE
        mae        = np.mean(abs_diff)
        # Threshold accuracies
        thresh     = np.maximum(pred_flat / target_flat, target_flat / pred_flat)
        delta1     = np.mean(thresh < 1.25)
        delta2     = np.mean(thresh < 1.25 ** 2)
        delta3     = np.mean(thresh < 1.25 ** 3)

        return {
            "abs_rel" : abs_rel,
            "sq_rel"  : sq_rel,
            "rmse"    : rmse,
            "rmse_log": rmse_log,
            "mae"     : mae,
            "delta1"  : delta1,
            "delta2"  : delta2,
            "delta3"  : delta3
        }

    def _print_results(self):
        """Print the measured results."""
        results = self._results

        message = ""
        # Headers
        for m, v in results.items():
            if v:
                message += f"{f'{m}':<10}\t"
        message += "\n"
        # Values
        for i, (m, v) in enumerate(results.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:.10f}\n"
                else:
                    message += f"{v:.10f}\t"
        print(f"{message}\n")

    # --- CLI ---
    @staticmethod
    def parse_args() -> box.Box:
        """Parse command line arguments.

        Returns:
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="metric_depth")
        parser.add_argument("--input-dir",   type=str, help="Input image directory.")
        parser.add_argument("--target-dir",  type=str, help="Ground-truth image directory.")
        parser.add_argument("--result-file", type=str, help="Result file.")
        parser.add_argument("--arch",        type=str, help="Model's architecture.")
        parser.add_argument("--model",       type=str, help="Model's fullname.")
        parser.add_argument("--data",        type=str, help="Source data name.")
        parser.add_argument("--device",      type=str, help="Running devices.")
        parser.add_argument("--imgsz",       type=int, default=512)
        parser.add_argument("--resize",      action="store_true")
        parser.add_argument("--normalize",   action="store_true")
        parser.add_argument("--use-color",   action="store_true")
        parser.add_argument("--save-txt",    action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        return box.Box(vars(parser.parse_args()))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    args   = DepthEvaluator.parse_args()
    runner = DepthEvaluator(args)
    runner.run()

# endregion
