#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image quality assessment (IQA) evaluator.

This module provides a runner for measuring IQA metrics.
"""

from __future__ import annotations

__all__ = [
    "IQAEvaluator",
]

import argparse
import logging

import albumentations as A
import box
import pyiqa
import pyiqa.default_model_configs
import pyiqa.models.inference_model

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
_METRICS     = pyiqa.default_model_configs.DEFAULT_CONFIGS


# ==============================================================================
# region CONTROL
# ==============================================================================

class IQAEvaluator:
    """A runner for measuring IQA metrics."""

    # --- Lifecycle & Initialization ---
    def __init__(self, cfg: box.Box):
        # Assign attributes
        self._cfg         = cfg
        self.verbose      = cfg.verbose
        self._arch        = cfg.arch
        self._model       = cfg.model
        self._data        = cfg.data
        self._device      = create_device(cfg.device)
        self._imgsz       = I.imgsz(cfg.imgsz)
        self._resize      = cfg.resize
        self._use_gt_mean = cfg.use_gt_mean
        self._save_txt    = cfg.save_txt

        # Resolve paths
        self._input_dir = Path(cfg.input_dir).normalize(exist=True)
        if cfg.target_dir:
            self._target_dir = Path(cfg.target_dir).normalize(exist=True)
        else:
            self._target_dir = self._input_dir.replace_part("image", "ref")

        # Setup components (lightweight)
        # Define metrics
        self._metrics   = self._create_metrics()
        self._metrics_f = self._create_metrics_func()

        # Initialize states
        self._results         = {}
        self._results_gt_mean = {}

    def _create_metrics(self) -> list[str]:
        """Create metrics."""
        metrics = self._cfg.metric
        metrics = list(_METRICS.names()) if ("all" in metrics or "*" in metrics) else metrics
        return metrics

    def _create_metrics_func(self) -> dict[str, callable]:
        """Create metric functions."""
        metric_f = {}
        for i, m in enumerate(self._metrics):
            if m in _METRICS:
                metric_f[m] = pyiqa.create_metric(metric_name=m, as_loss=False, device=self._device)
        return metric_f

    def _build_dataloader(self) -> DataLoader:
        """Create a dataloader for the given dataset."""
        h, w      = self._imgsz
        transform = A.Compose([
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), normalization="min_max"),
            A.ToTensorV2(transpose_mask=True),
        ])
        if self._resize:
            transform = A.Resize(height=h, width=w) + transform

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
        self._results = self._measure(dataloader=dataloader, use_gt_mean=False)

        if self._use_gt_mean:
            self._results_gt_mean = self._measure(dataloader=dataloader, use_gt_mean=True)

        # Print results
        self._print_results()

    def _measure(self, dataloader: DataLoader, use_gt_mean: bool = False) -> dict:
        """Measure IQA metrics based on the configuration."""
        # Resolve attributes
        model    = self._model
        data     = self._data
        device   = self._device
        metrics  = self._metrics
        metric_f = self._metrics_f
        verbose  = self.verbose

        # Prepare containers
        values   = {m: [] for m in metrics}
        results  = {}

        # Processing loop
        with create_progress_bar(transient=not verbose) as pbar:
            desc = f"[bright_yellow]Measuring {model} | {data} (GT Mean)" if use_gt_mean else f"[bright_yellow]Measuring {model} | {data}"
            for i, datapoint in pbar.track(
                sequence    = enumerate(dataloader),
                total       = len(dataloader),
                description = desc
            ):
                image  = datapoint["image"]
                target = datapoint.get("target")
                if target in [ [], [None], None ]:
                    target = None
                elif image.shape != target.shape:
                    image  = image.permute(0, 1, 3, 2)

                # Move tensors to a device
                image  =  image.to(device=device)
                target = target.to(device=device) if target is not None else None

                # Measure metric
                for m in metrics:
                    if target is None and _METRICS[m]["metric_mode"] == "FR":
                        continue
                    elif target is not None and _METRICS[m]["metric_mode"] == "FR":
                        values[m].append(metric_f[m](image, target))
                    else:
                        values[m].append(metric_f[m](image))

        # Aggregate results
        for m, v in values.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None

        return results

    def _print_results(self):
        """Print the measured results."""
        results         = self._results
        results_gt_mean = self._results_gt_mean

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
        for i, (m, v) in enumerate(results_gt_mean.items()):
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
        parser = argparse.ArgumentParser(description="metric_iqa")
        parser.add_argument("--input-dir",   type=str, help="Input image directory.")
        parser.add_argument("--target-dir",  type=str, help="Ground-truth image directory.")
        parser.add_argument("--result-file", type=str, help="Result file.")
        parser.add_argument("--arch",        type=str, help="Model's architecture.")
        parser.add_argument("--model",       type=str, help="Model's fullname.")
        parser.add_argument("--data",        type=str, help="Source data name.")
        parser.add_argument("--device",      type=str, help="Running devices.")
        parser.add_argument("--imgsz",       type=int, default=512)
        parser.add_argument("--resize",      action="store_true")
        parser.add_argument("--metric",      type=str, action="append", help="Measuring metric.")
        parser.add_argument("--use-gt-mean", action="store_true")
        parser.add_argument("--save-txt",    action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        return box.Box(vars(parser.parse_args()))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    args   = IQAEvaluator.parse_args()
    runner = IQAEvaluator(args)
    runner.run()

# endregion
