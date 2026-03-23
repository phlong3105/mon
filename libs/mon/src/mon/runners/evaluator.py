#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluators.

This module provides several metric evaluators.
"""

from __future__ import annotations

__all__ = [
    "DQAEvaluator",
    "IQAEvaluator",
]

import argparse
import logging
from typing import override

import matplotlib
import numpy as np
import pyiqa
import pyiqa.default_model_configs
import pyiqa.models.inference_model

from mon.core import (
    console,
    create_progress_bar,
    DeviceLike,
    log_error,
    Path,
    PathLike,
    Size,
    SizeLike,
)
from mon.dataset import DataLoader, IQADataset, transform as T
from mon.metrics import compute_depth_metrics
from .base import Evaluator

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region IQA
# ==============================================================================

class IQAEvaluator(Evaluator):
    """A runner for measuring image quality evaluation (IQA) metrics."""

    all_metrics = pyiqa.default_model_configs.DEFAULT_CONFIGS

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: PathLike,
        target_dir: PathLike | None,
        result_file: PathLike | None,
        arch: str,
        model: str,
        data: str,
        metrics: list[str],
        device: DeviceLike,
        imgsz: SizeLike = 512,
        resize: bool = False,
        use_gt_mean: bool = False,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (PathLike): Input image directory.
            target_dir (PathLike | None): Ground-truth image directory. If None,
                it will be inferred from ``input_dir``.
            result_file (PathLike | None): Result file. If None, results will
                not be saved.
            arch (str): Model's architecture.
            model (str): Model's fullname.
            data (str): Source data name.
            metrics (list[str]): List of metrics to measure.
            device (DeviceLike): Running device.
            imgsz (SizeLike, optional): Image size for resizing. If resize is
                False, this will be ignored. Defaults to 512.
            resize (bool, optional): Whether to resize images to ``imgsz``
                before measuring metrics. Defaults to False.
            use_gt_mean (bool, optional): Whether to use the mean of ground-truth
                images as the reference for NR metrics. Defaults to False.
            verbose (bool, optional): Verbose mode. Defaults to True.
        """
        super().__init__(
            input_dir=input_dir,
            target_dir=target_dir,
            result_file=result_file,
            metrics=metrics,
            device=device,
            verbose=verbose
        )

        # Assign attributes
        self.arch = arch
        self.model = model
        self.data = data
        self.imgsz = imgsz
        self.resize = resize
        self.use_gt_mean = use_gt_mean

        # Allocate resources
        self._results_gt_mean = {}

    @override
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` and ``self._metrics_func`` attributes.``"""
        _metrics = []
        _metrics_func = {}
        for i, m in enumerate(metrics):
            if m in self.all_metrics:
                _metrics.append(m)
                _metrics_func[m] = pyiqa.create_metric(
                    metric_name=m, as_loss=False, device=self.device
                )
            else:
                log_error(f"Unsupported metric: {m}. Skipping...")

        self._metrics = _metrics
        self._metrics_func = _metrics_func

    @override
    def _init_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])
        if self.resize:
            h, w = self.imgsz.hw
            transforms = T.Resize(height=h, width=w) + transforms

        return DataLoader(
            dataset=IQADataset(
                input_dir=self.input_dir,
                target_dir=self.target_dir,
                transforms=transforms,
                verbose=False,
            ),
            batch_size=1,
        )

    # --- Properties ---
    @property
    def imgsz(self) -> Size:
        """Return the image size for resizing."""
        return self._imgsz

    @imgsz.setter
    def imgsz(self, value: SizeLike):
        """Set the image size for resizing."""
        self._imgsz = Size.from_value(value)

    @property
    def results_gt_mean(self) -> dict:
        """Return the dictionary of measured results with ground-truth mean."""
        return self._results_gt_mean

    # --- Creation ---
    @classmethod
    def from_cli(cls) -> "IQAEvaluator":
        """Create an instance of IQAEvaluator from command-line arguments."""
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
        parser.add_argument("--verbose",     action="store_true")
        args = vars(parser.parse_args())
        return cls(**args)

    # --- Measure ---
    @override
    def measure(self):
        """Run the metric measurement process."""
        # Summarize the current run
        self.log_summary()

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        dataloader = self._init_dataloader()

        # Processing
        self._results = self._measure(dataloader=dataloader, use_gt_mean=False)

        if self.use_gt_mean:
            self._results_gt_mean = self._measure(dataloader=dataloader, use_gt_mean=True)

        # Print results
        self.log_results()

    def _measure(self, dataloader: DataLoader, use_gt_mean: bool = False) -> dict:
        """Measure IQA metrics based on the configuration.

        Args:
            dataloader (DataLoader): The dataloader for the dataset.
            use_gt_mean (bool, optional): Whether to use the mean of ground-truth
                images as the reference for NR metrics. Defaults to False.
        """
        # Resolve attributes
        model = self.model
        data = self.data
        device = self.device
        metrics = self.metrics
        metrics_func = self.metrics_func
        verbose = self.verbose

        # Processing loop
        values = {m: [] for m in metrics}

        with create_progress_bar(transient=not verbose) as pbar:
            if use_gt_mean:
                desc = f"[bright_yellow]Measuring {model} | {data} (GT Mean)"
            else:
                desc = f"[bright_yellow]Measuring {model} | {data}"

            for i, datapoint in pbar.track(
                sequence=enumerate(dataloader),
                total=len(dataloader),
                description=desc,
            ):
                image = datapoint["image"]
                target = datapoint.get("target", None)

                # Sometimes image and target may have different orientations
                # (H, W) vs (W, H). We check the image and target sizes and
                # transpose the image if needed.
                image_sz = image.shape[-2:]
                if target is not None:
                    target_sz = target.shape[-2:]
                    if image_sz[0] == target_sz[1]:
                        image = image.transpose(2, 3)

                # Move tensors to a device
                image = image.to(device=device)
                target = target.to(device=device) if target is not None else None

                # Measure metric
                for m in metrics:
                    if target is None and self.all_metrics[m]["metric_mode"] == "FR":
                        continue
                    elif target is not None and self.all_metrics[m]["metric_mode"] == "FR":
                        values[m].append(metrics_func[m](image, target))
                    else:
                        values[m].append(metrics_func[m](image))

        # Aggregate results
        results = {}
        for m, v in values.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None
        return results

    # --- Logging ---
    def log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True
        console.rule(f"[bold red] {self.model}")
        console.log(f"[bold green]Model : {self.model}")
        console.log(f"[bold red]Data  : {self.data}")
        console.log(f"[bold]Device: {self.device}")

    def log_results(self):
        """Print the measured results."""
        results = self.results
        results_gt_mean = self.results_gt_mean

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

# endregion


# ==============================================================================
# region DQA
# ==============================================================================

class DQAEvaluator(Evaluator):
    """A runner for measuring depth quality evaluation (DQA) metrics."""

    all_metrics = [
        "abs_rel", "sq_rel", "rmse", "rmse_log", "mae",
        "delta1", "delta2", "delta3",
    ]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: PathLike,
        target_dir: PathLike | None,
        result_file: PathLike | None,
        arch: str,
        model: str,
        data: str,
        metrics: list[str],
        device: DeviceLike,
        imgsz: SizeLike = 512,
        resize: bool = False,
        normalize: bool = False,
        use_color: bool = False,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (PathLike): Input image directory.
            target_dir (PathLike | None): Ground-truth image directory. If None,
                it will be inferred from ``input_dir``.
            result_file (PathLike | None): Result file. If None, results will
                not be saved.
            arch (str): Model's architecture.
            model (str): Model's fullname.
            data (str): Source data name.
            metrics (list[str]): List of metrics to measure.
            device (DeviceLike): Running device.
            imgsz (SizeLike, optional): Image size for resizing. If resize is
                False, this will be ignored. Defaults to 512.
            resize (bool, optional): Whether to resize images to ``imgsz``
                before measuring metrics. Defaults to False.
            normalize (bool, optional): Whether to normalize depth maps before
                measuring metrics. Defaults to False.
            use_color (bool, optional): Whether to convert depth maps to color
                images before measuring metrics. Defaults to False.
            verbose (bool, optional): Verbose mode. Defaults to True.
        """
        super().__init__(
            input_dir=input_dir,
            target_dir=target_dir,
            result_file=result_file,
            metrics=metrics,
            device=device,
            verbose=verbose
        )

        # Assign attributes
        self.arch = arch
        self.model = model
        self.data = data
        self.imgsz = imgsz
        self.resize = resize
        self.normalize = normalize
        self.use_color = use_color

        # Allocate resources
        self._cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    @override
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` and ``self._metrics_func`` attributes.``"""
        self._metrics = [m.lower() for m in self.all_metrics]
        self._metrics_func = {}

    @override
    def _init_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        if self.resize:
            h, w = self.imgsz.hw
            transforms = T.Compose([T.Resize(height=h, width=w)])
        else:
            transforms = None

        return DataLoader(
            dataset=IQADataset(
                input_dir=self.input_dir,
                target_dir=self.target_dir,
                transforms=transforms,
                verbose=False,
            ),
            batch_size=1,
        )

    # --- Properties ---
    @property
    def imgsz(self) -> Size:
        """Return the image size for resizing."""
        return self._imgsz

    @imgsz.setter
    def imgsz(self, value: SizeLike):
        """Set the image size for resizing."""
        self._imgsz = Size.from_value(value)

    # --- Creation ---
    @classmethod
    def from_cli(cls) -> "DQAEvaluator":
        """Create an instance of IQAEvaluator from command-line arguments."""
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
        parser.add_argument("--normalize",   action="store_true")
        parser.add_argument("--use-color",   action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        args = vars(parser.parse_args())
        return cls(**args)

    # --- Measure ---
    @override
    def measure(self):
        """Run the metric measurement process."""
        # Summarize the current run
        self.log_summary()

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        dataloader = self._init_dataloader()

        # Processing
        self._results = self._measure(dataloader=dataloader)

        # Print results
        self.log_results()

    def _measure(self, dataloader: DataLoader) -> dict:
        """Measure IQA metrics based on the configuration.

        Args:
            dataloader (DataLoader): The dataloader for the dataset.
        """
        # Resolve attributes
        model = self.model
        data = self.data
        metrics = self.metrics
        cmap = self._cmap
        use_color = self.use_color
        verbose = self.verbose

        # Processing loop
        values = {m: [] for m in metrics}

        with create_progress_bar(transient=not verbose) as pbar:
            desc = f"[bright_yellow]Measuring {model} | {data}"
            for i, datapoint in pbar.track(
                sequence=enumerate(dataloader),
                total=len(dataloader),
                description=desc,
            ):
                image = datapoint["image"]
                target = datapoint["target"]

                # Sometimes image and target may have different orientations
                # (H, W) vs (W, H). We check the image and target sizes and
                # transpose the image if needed.
                image_sz = image.shape[-2:]
                if target is not None:
                    target_sz = target.shape[-2:]
                    if image_sz[0] == target_sz[1]:
                        image = image.transpose(2, 3)

                if use_color:
                    image = (cmap(image)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    target = (cmap(target)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                # Measure metric
                measured_results = compute_depth_metrics(image, target)
                for k, v in measured_results.items():
                    if k in values:
                        values[k].append(v)

        # Aggregate results
        results = {}
        for m, v in values.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None
        return results

    # --- Logging ---
    def log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True
        console.rule(f"[bold red] {self.model}")
        console.log(f"[bold green]Model : {self.model}")
        console.log(f"[bold red]Data  : {self.data}")
        console.log(f"[bold]Device: {self.device}")

    def log_results(self):
        """Print the measured results."""
        results = self.results

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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    iqa = IQAEvaluator.from_cli()
    iqa.measure()

# endregion
