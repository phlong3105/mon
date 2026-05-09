#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluators.

This module provides several metric evaluators.
"""

from __future__ import annotations

__all__ = [
    "COCOEvaluator",
    "DQAEvaluator",
    "Evaluator",
    "IQAEvaluator",
    "InstanceIQAEvaluator",
]

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, override

import cv2
import matplotlib
import numpy as np
import pyiqa
import pyiqa.default_model_configs
import pyiqa.models.inference_model
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor

from mon import BBoxFormat
from mon.core import (
    console,
    create_progress_bar,
    log_error,
    METRICS,
    Path,
    Size,
    sys_ctx,
)
from mon.dataset import DataLoader, IQADataset, transform as T
from mon.metrics import compute_depth_metrics
from mon.ops import convert_labels_to_json

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Evaluator(ABC):
    """Base class for all evaluators."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        metrics: list[str],
        device: torch.device | str | int = torch.device("cpu"),
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            metrics (list[str]): The list of metrics to evaluate.
            device (torch.device | str | int): Running device. Defaults to "cpu".
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Assign attributes
        self.verbose = verbose
        self.device = device

        # Allocate resources
        self._metrics: dict[str, dict] = []
        self._init_metrics(metrics)

    @abstractmethod
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` attributes."""
        pass

    # --- Properties ---
    @property
    def device(self) -> torch.device:
        """Return the device to use."""
        return self._device

    @device.setter
    def device(self, device: torch.device | str | int):
        """Set the device to use."""
        self._device = sys_ctx.get_torch_device(device)

    # --- Measure ---
    @abstractmethod
    def measure(self) -> dict[str, Any]:
        """Run the evaluation.

        Returns:
            dict[str, Any]: A dictionary containing the evaluation results.
        """
        pass

    @abstractmethod
    def _build_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        pass

    # --- Logging ---
    @abstractmethod
    def _log_summary(self):
        """Log a summary of the current run."""
        pass

    @abstractmethod
    def _log_results(self, results: dict[str, Any]):
        """Log the evaluation results.

        Args:
            results (dict[str, Any]): The evaluation results to log.
        """
        pass

# endregion


# ==============================================================================
# region IQA
# ==============================================================================

class IQAEvaluator(Evaluator):
    """A runner for measuring image quality evaluation (IQA) metrics."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path | None,
        arch: str,
        model: str,
        data: str,
        metrics: list[str],
        device: torch.device | str | int = torch.device("cpu"),
        imgsz: Size = (512, 512),
        resize: bool = False,
        use_gt_mean: bool = False,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (Path): Input image directory.
            target_dir (Path | None): Ground-truth image directory.
                If None, it will be inferred from ``input_dir``.
            arch (str): Model's architecture.
            model (str): Model's fullname.
            data (str): Source data name.
            metrics (list[str]): List of metrics to measure.
            device (torch.device | str | int): Running device. Defaults to "cpu".
            imgsz (Size, optional): Image size for resizing. If resize is False,
                this will be ignored. Defaults to 512.
            resize (bool, optional): Whether to resize images to ``imgsz``
                before measuring metrics. Defaults to False.
            use_gt_mean (bool, optional): Whether to use the mean of ground-truth
                images as the reference for NR metrics. Defaults to False.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(metrics=metrics, device=device, verbose=verbose)

        # Assign attributes
        self._input_dir = Path(input_dir).normalize()
        self._target_dir = Path(target_dir).normalize() if target_dir else None

        self._arch = arch
        self._model = model
        self._data = data
        self._imgsz = Size.from_any(imgsz)
        self._resize = resize
        self._use_gt_mean = use_gt_mean

    @override
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` attributes."""
        _metrics: dict[str, dict] = {}

        for i, m in enumerate(metrics):
            if m in pyiqa.default_model_configs.DEFAULT_CONFIGS:
                func = pyiqa.create_metric(metric_name=m, as_loss=False, device=self.device)
            elif m in METRICS:
                func = METRICS.build(name=m, device=self.device)
            else:
                log_error(f"unsupported metric {m}, skipping...")
                func = None

            _metrics[m] = {
                "func": func,
                "metric_mode": METRICS[m]["metric_mode"],
                "lower_better": METRICS[m]["lower_better"],
                "score_range": METRICS[m]["score_range"],
            }

        self._metrics = _metrics

    # --- Creation ---
    @classmethod
    def from_cli(cls, **kwargs) -> "IQAEvaluator":
        """Create an instance of IQAEvaluator from command-line arguments."""
        parser = argparse.ArgumentParser(description="metric_iqa")
        parser.add_argument("--input-dir",   type=str, help="Input image directory.")
        parser.add_argument("--target-dir",  type=str, help="Ground-truth image directory.")
        parser.add_argument("--result-file", type=str, help="Result file.")
        parser.add_argument("--arch",        type=str, help="Model's architecture.")
        parser.add_argument("--model",       type=str, help="Model's fullname.")
        parser.add_argument("--data",        type=str, help="Source data name.")
        parser.add_argument("--metric",      type=str, action="append", help="Measuring metric.")
        parser.add_argument("--device",      type=str, help="Running devices.")
        parser.add_argument("--imgsz",       type=int, default=512)
        parser.add_argument("--resize",      action="store_true")
        parser.add_argument("--use-gt-mean", action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        args = vars(parser.parse_args())
        args["metrics"] = args.pop("metric")  # Rename "metric" to "metrics"
        args |= kwargs  # Override with additional kwargs
        return cls(**args)

    # --- Measure ---
    @override
    def measure(self) -> dict[str, float]:
        """Run the metric measurement process.

        Returns:
            dict[str, Any]: A dictionary containing the evaluation results.
        """
        # Summarize the current run
        self._log_summary()

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        dataloader = self._build_dataloader()

        # Processing
        results = self._measure(dataloader=dataloader, use_gt_mean=False)

        results_gt_mean = None
        if self._use_gt_mean:
            results_gt_mean = self._measure(dataloader=dataloader, use_gt_mean=True)

        # Print results
        self._log_results(results=results, results_gt_mean=results_gt_mean)

        # Return results
        return results

    def _measure(self, dataloader: DataLoader, use_gt_mean: bool = False) -> dict[str, float]:
        """Measure IQA metrics based on the configuration.

        Args:
            dataloader (DataLoader): The dataloader for the dataset.
            use_gt_mean (bool, optional): Whether to use the mean of ground-truth
                images as the reference for NR metrics. Defaults to False.

        Returns:
            dict[str, float]: The dictionary of measured results.
        """
        # Resolve attributes
        model = self._model
        data = self._data
        metrics = self._metrics
        device = self.device
        verbose = self.verbose

        # Processing loop
        results = {m: [] for m in metrics.keys()}

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
                datapoint = datapoint.to(device)
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

                # Measure metric
                for m in metrics:
                    if target is None and metrics[m]["metric_mode"] == "FR":
                        continue

                    func = metrics[m]["func"]
                    if metrics[m]["metric_mode"] == "FR":
                        results[m].append(func(image, target).item())
                    else:
                        results[m].append(func(image).item())

        # Aggregate results
        for m, v in results.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None
        return results

    @override
    def _build_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])
        if self._resize:
            h, w = self._imgsz.hw
            transforms = T.Resize(height=h, width=w) + transforms

        return DataLoader(
            dataset=IQADataset(
                input_dir=self._input_dir,
                target_dir=self._target_dir,
                transforms=transforms,
                verbose=False,
            ),
            batch_size=1,
        )

    # --- Logging ---
    @override
    def _log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True

        console.rule(f"[bold red] {self._model}")
        console.log(f"[bold green]Model : {self._model}")
        console.log(f"[bold red]Data  : {self._data}")
        console.log(f"[bold]Device: {self._device}")

    @override
    def _log_results(
        self,
        results: dict[str, float],
        results_gt_mean: dict[str, float] | None = None,
    ):
        """Log the evaluation results.

        Args:
            results (dict[str, float]): The evaluation results to log.
            results_gt_mean (dict[str, float] | None, optional): The evaluation
                results using GT mean as reference. If None, it will not be logged.
                Defaults to None.
        """
        pad = 10

        # Headers
        header = ""
        for m, v in results.items():
            if v:
                header += f"{f'{m}':<{pad}}\t"

        # Values
        message = ""
        for i, (m, v) in enumerate(results.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:.{pad}f}\n"
                else:
                    message += f"{v:.{pad}f}\t"
        for i, (m, v) in enumerate(results_gt_mean.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:.{pad}f}\n"
                else:
                    message += f"{v:.{pad}f}\t"

        print(f"{header}")
        print(f"{message}")


class InstanceIQAEvaluator(Evaluator):
    """A runner for measuring image quality evaluation (IQA) metrics of a
    single image.
    """

    all_metrics = pyiqa.default_model_configs.DEFAULT_CONFIGS
    excluded_stems = ["image", "target", "depth"]
    target_stem = "target"

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: Path,
        metrics: list[str],
        device: torch.device | str | int = torch.device("cpu"),
        imgsz: Size = (512, 512),
        resize: bool = False,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (Path): Input image directory.
            metrics (list[str]): List of metrics to measure.
            device (torch.device | str | int): Running device. Defaults to "cpu".
            imgsz (Size, optional): Image size for resizing. If resize is False,
                this will be ignored. Defaults to 512.
            resize (bool, optional): Whether to resize images to ``imgsz``
                before measuring metrics. Defaults to False.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(metrics=metrics, device=device, verbose=verbose)

        # Assign attributes
        self._input_dir = Path(input_dir).normalize()
        self._imgsz = Size.from_any(imgsz)
        self._resize = resize

    @override
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` attributes."""
        _metrics: dict[str, dict] = {}

        for i, m in enumerate(metrics):
            if m in pyiqa.default_model_configs.DEFAULT_CONFIGS:
                func = pyiqa.create_metric(metric_name=m, as_loss=False, device=self.device)
            elif m in METRICS:
                func = METRICS.build(name=m, device=self.device)
            else:
                log_error(f"unsupported metric {m}, skipping...")
                func = None

            _metrics[m] = {
                "func": func,
                "metric_mode": METRICS[m]["metric_mode"],
                "lower_better": METRICS[m]["lower_better"],
                "score_range": METRICS[m]["score_range"],
            }

        self._metrics = _metrics

    # --- Creation ---
    @classmethod
    def from_cli(cls, **kwargs) -> "IQAEvaluator":
        """Create an instance of IQAEvaluator from command-line arguments."""
        parser = argparse.ArgumentParser(description="metric_iqa")
        parser.add_argument("--input-dir",   type=str, help="Input image directory.")
        parser.add_argument("--result-file", type=str, help="Result file.")
        parser.add_argument("--metric",      type=str, action="append", help="Measuring metric.")
        parser.add_argument("--device",      type=str, help="Running devices.")
        parser.add_argument("--imgsz",       type=int, default=512)
        parser.add_argument("--resize",      action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        args = vars(parser.parse_args())
        args["metrics"] = args.pop("metric")  # Rename "metric" to "metrics"
        args |= kwargs  # Override with additional kwargs
        return cls(**args)

    # --- Measure ---
    @override
    def measure(self) -> dict[str, float]:
        """Run the metric measurement process.

        Returns:
            dict[str, float]: A dictionary containing the evaluation results.
        """
        # Summarize the current run
        self._log_summary()

        # Processing
        results = self._measure()

        # Print results
        self._log_results(results=results)

        # Return results
        return results

    @torch.inference_mode()
    def _measure(self) -> dict[str, float]:
        """Measure IQA metrics based on the configuration.

        Returns:
            dict[str, float]: The dictionary of measured results.
        """
        # Resolve attributes
        metrics = self._metrics
        device = self.device
        verbose = self.verbose

        # Define images and target
        image_files = sorted(list(self._input_dir.rglob(f"*")))
        image_files = [i for i in image_files if i.stem not in self.excluded_stems]
        image_files = [i for i in image_files if i.is_image_file(exists=True)]

        target_file = self._input_dir / self.target_stem
        target_file = target_file.image_file
        if target_file.is_image_file(exists=True):
            target = cv2.imread(str(target_file))
        else:
            target = None

        # Define transforms
        transforms: T.Compose = self._build_transforms()

        # Processing loop
        results = {i.stem: [] for i in image_files}

        with create_progress_bar(transient=not verbose) as pbar:
            for i, image_file in pbar.track(
                sequence=enumerate(image_files),
                total=len(image_files),
                description=f"[bright_yellow]Measuring",
            ):
                # Read image file
                image = cv2.imread(str(image_file))

                # Transform image and target
                transformed = transforms(image=image, target=target)
                image_t = transformed["image"].unsqueeze(0).to(device)
                target_t = transformed["target"].unsqueeze(0).to(device) if target is not None else None

                # Sometimes image and target may have different orientations
                # (H, W) vs (W, H). We check the image and target sizes and
                # transpose the image if needed.
                image_sz = image_t.shape[-2:]
                if isinstance(target_t, Tensor):
                    target_sz = target_t.shape[-2:]
                    if image_sz[0] == target_sz[1]:
                        image_t = image_t.transpose(2, 3)

                # Measure metric
                values = {m: [] for m in metrics.keys()}
                for m in metrics:
                    if target is None and metrics[m]["metric_mode"] == "FR":
                        continue

                    func = metrics[m]["func"]
                    if metrics[m]["metric_mode"] == "FR":
                        values[m] = func[m](image_t, target_t).item()
                    else:
                        values[m] = func[m](image_t).item()

                # Aggregate results
                results[image_file.stem] = values

        return results

    def _build_transforms(self) -> T.Compose:
        """Build a dataloader for the given dataset."""
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ], additional_targets={"target": "image"})
        if self._resize:
            h, w = self._imgsz.hw
            transforms = T.Resize(height=h, width=w) + transforms

        return transforms

    @override
    def _build_dataloader(self):
        """Build a dataloader for the given dataset."""
        pass

    # --- Logging ---
    @override
    def _log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True

        console.rule(f"[bold red] IQA Metric")
        console.log(f"[bold]Data  : {self._input_dir.name}")
        console.log(f"[bold]Device: {self._device}")

    @override
    def _log_results(self, results: dict[str, Any]):
        """Print the measured results.

        Args:
            results (dict[str, float]): The evaluation results to log.
        """
        pad = 7

        # Headers
        first_item: dict = list(results.values())[0]
        header = f"{f'Model':<{pad * 4}}\t"
        for m, v in first_item.items():
            header += f"{f'{m}':<{pad}}\t"
        header += "\n"
        header += "-" * ((pad + 4) * (len(first_item) + 1))

        # Values
        message = ""
        for i, (model, values) in enumerate(results.items()):
            message += f"{f'{model}':<{pad * 4}}\t"
            for k, v in values.items():
                message += f"{f'{v:6.4f}':<{pad}}\t"
            message += "\n"

        print(f"{header}")
        print(f"{message}")

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
        input_dir: Path,
        target_dir: Path,
        arch: str,
        model: str,
        data: str,
        device: torch.device | str | int = torch.device("cpu"),
        imgsz: Size = (512, 512),
        resize: bool = False,
        normalize: bool = False,
        use_color: bool = False,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (Path): Input image directory.
            target_dir (Path): Ground-truth image directory.
            arch (str): Model's architecture.
            model (str): Model's fullname.
            data (str): Source data name.
            device (torch.device | str | int): Running device. Defaults to "cpu".
            imgsz (Size, optional): Image size for resizing. If resize is False,
                this will be ignored. Defaults to 512.
            resize (bool, optional): Whether to resize images to ``imgsz``
                before measuring metrics. Defaults to False.
            normalize (bool, optional): Whether to normalize depth maps before
                measuring metrics. Defaults to False.
            use_color (bool, optional): Whether to convert depth maps to color
                images before measuring metrics. Defaults to False.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(metrics=[], device=device, verbose=verbose)

        # Assign attributes
        self._input_dir = Path(input_dir).normalize()
        self._target_dir = Path(target_dir).normalize() if target_dir else None
        self._arch = arch
        self._model = model
        self._data = data
        self._imgsz = Size.from_any(imgsz)
        self._resize = resize
        self._normalize = normalize
        self._use_color = use_color

        # Allocate resources
        self._cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    @override
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` and ``self._metrics_func`` attributes.``"""
        self._metrics = [m.lower() for m in self.all_metrics]
        self._metrics_func = {}

    # --- Creation ---
    @classmethod
    def from_cli(cls, **kwargs) -> "DQAEvaluator":
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
        args |= kwargs  # Override with additional kwargs
        return cls(**args)

    # --- Measure ---
    @override
    def measure(self) -> dict[str, float]:
        """Run the metric measurement process.

        Returns:
            dict[str, float]: A dictionary containing the evaluation results.
        """
        # Summarize the current run
        self._log_summary()

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        dataloader = self._build_dataloader()

        # Processing
        results = self._measure(dataloader=dataloader)

        # Print results
        self._log_results(results=results)

        # Return results
        return results

    def _measure(self, dataloader: DataLoader) -> dict[str, float]:
        """Measure IQA metrics based on the configuration.

        Args:
            dataloader (DataLoader): The dataloader for the dataset.

        Returns:
            dict[str, float]: The dictionary of measured results.
        """
        # Resolve attributes
        model = self._model
        data = self._data
        metrics = self._metrics
        cmap = self._cmap
        use_color = self._use_color
        verbose = self.verbose

        # Processing loop
        results = {m: [] for m in metrics}

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
                    if k in results:
                        results[k].append(v)

        # Aggregate results
        for m, v in results.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None
        return results

    @override
    def _build_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        if self._resize:
            h, w = self._imgsz.hw
            transforms = T.Compose([T.Resize(height=h, width=w)])
        else:
            transforms = None

        return DataLoader(
            dataset=IQADataset(
                input_dir=self._input_dir,
                target_dir=self._target_dir,
                transforms=transforms,
                verbose=False,
            ),
            batch_size=1,
        )

    # --- Logging ---
    @override
    def _log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True

        console.rule(f"[bold red] {self._model}")
        console.log(f"[bold green]Model : {self._model}")
        console.log(f"[bold red]Data  : {self._data}")
        console.log(f"[bold]Device: {self._device}")

    @override
    def _log_results(self, results: dict[str, Any]):
        """Print the measured results.

        Args:
            results (dict[str, float]): The evaluation results to log.
        """
        pad = 10
        message = ""

        # Headers
        for m, v in results.items():
            if v:
                message += f"{f'{m}':<{pad}}\t"
        message += "\n"
        # Values
        for i, (m, v) in enumerate(results.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:.{pad}f}\n"
                else:
                    message += f"{v:.{pad}f}\t"

        print(f"{message}\n")

# endregion


# ==============================================================================
# region COCO
# ==============================================================================

class COCOEvaluator(Evaluator):
    """A runner for measuring COCO metrics."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        image_dir: Path | None,
        input_dir: Path | None,
        target_dir: Path | None,
        input_json: Path | None,
        target_json: Path | None,
        remap: Path | None,
        arch: str,
        model: str,
        data: str,
        fmt: BBoxFormat = BBoxFormat.CXCYWHN,
        device: torch.device | str | int = torch.device("cpu"),
        exist_ok: bool = False,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            image_dir (Path): Input image directory.
            input_dir (Path | None): Input annotation directory.
                If None, ``input_json`` must be provided.
            target_dir (Path | None): Ground-truth annotation directory.
                If None, ``target_json`` must be provided.
            input_json (Path | None): Input JSON file.
                If None, ``input_ann_dir`` must be provided.
            target_json (Path | None): Ground-truth JSON file.
                If None, ``target_ann_dir`` must be provided.
            remap (Path | None): Classes re-map definition file.
                If None, no re-mapping will be applied.
            arch (str): Model's architecture.
            model (str): Model's fullname.
            data (str): Source data name.
            fmt (BBoxFormat, optional): Bounding box format. Defaults to BBoxFormat.CXCYWHN.
            device (torch.device | str | int): Running device.
            exist_ok (bool, optional): Whether to overwrite the existing JSON files.
                Defaults to False.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(metrics=[], device=device, verbose=verbose)

        # Assign attributes
        self._arch = arch
        self._model = model
        self._data = data
        self._fmt = BBoxFormat(fmt)
        self._remap = Path(remap).normalize() if remap else None
        self._exist_ok = exist_ok

        # Resolve paths
        if input_json and exist_ok:
            input_json = Path(input_json).normalize()
        elif image_dir and input_dir:
            image_dir = Path(image_dir).normalize()
            input_dir = Path(input_dir).normalize()
            input_json = input_dir.parent / f"{input_dir.stem}.json"
        else:
            raise RuntimeError(
                f"Either input_json or (image_dir and input_dir) must be provided, got "
                f"input_json={input_json}\n"
                f"image_dir={image_dir}\n"
                f"input_dir={input_dir}"
            )

        if target_json and exist_ok:
            target_json = Path(target_json).normalize()
        elif image_dir and target_dir:
            target_dir = Path(target_dir).normalize()
            target_json = target_dir.parent / f"{target_dir.stem}.json"
        else:
            raise RuntimeError(
                f"Either target_json or (image_dir and target_dir) must be provided, got "
                f"target_json={target_json}\n"
                f"image_dir={image_dir}\n"
                f"target_dir={target_dir}"
            )

        self._image_dir = image_dir
        self._input_dir = input_dir
        self._target_dir = target_dir
        self._input_json = input_json
        self._target_json = target_json

    @override
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` attributes."""
        pass

    # --- Creation ---
    @classmethod
    def from_cli(cls, **kwargs) -> "COCOEvaluator":
        """Create an instance of IQAEvaluator from command-line arguments."""
        parser = argparse.ArgumentParser(description="metric_coco")
        parser.add_argument("--image-dir",   type=str, help="Input image directory.")
        parser.add_argument("--input-dir",   type=str, help="Input annotation directory.")
        parser.add_argument("--target-dir",  type=str, help="Ground-truth annotation directory.")
        parser.add_argument("--input-json",  type=str, help="Input JSON file.")
        parser.add_argument("--target-json", type=str, help="Ground-truth JSON file.")
        parser.add_argument("--remap",       type=str, help="Classes re-map definition file.")
        parser.add_argument("--arch",        type=str, help="Model's architecture.")
        parser.add_argument("--model",       type=str, help="Model's fullname.")
        parser.add_argument("--data",        type=str, help="Source data name.")
        parser.add_argument("--fmt",         choices=["coco", "voc", "yolo"], default="yolo")
        parser.add_argument("--device",      type=str, help="Running devices.")
        parser.add_argument("--exist-ok",    action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        args = vars(parser.parse_args())
        args |= kwargs  # Override with additional kwargs
        return cls(**args)

    # --- Measure ---
    @override
    def measure(self) -> dict[str, float]:
        """Run the metric measurement process.

        Returns:
            dict[str, Any]: A dictionary containing the evaluation results.
        """
        # Summarize the current run
        self._log_summary()

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        self._build_dataloader()

        # Processing
        coco_gt = COCO(self._target_json.as_posix())
        coco_dt = coco_gt.loadRes(self._input_json.as_posix())

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
        # coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = {
            "AP"    : coco_eval.stats[0],
            "AP50"  : coco_eval.stats[1],
            "AP75"  : coco_eval.stats[2],
            "APs"   : coco_eval.stats[3],
            "APm"   : coco_eval.stats[4],
            "APl"   : coco_eval.stats[5],
            "AR@1"  : coco_eval.stats[6],
            "AR@10" : coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
            "ARs"   : coco_eval.stats[9],
            "ARm"   : coco_eval.stats[10],
            "ARl"   : coco_eval.stats[11],
        }

        # Print results
        self._log_results(results=results)

        # Return results
        return results

    @override
    def _build_dataloader(self):
        """Build a dataloader for the given dataset."""
        # Convert label files to a COCO JSON file
        if not self._exist_ok:
            if self._input_json:
                self._input_json.unlink(missing_ok=True)
            if self._target_json:
                self._target_json.unlink(missing_ok=True)

        if not self._input_json.exists():
            convert_labels_to_json(
                image_dir=self._image_dir,
                label_dir=self._input_dir,
                output_json=self._input_json,
                fmt=self._fmt,
                remap=self._remap,
            )
        if not self._target_json.exists():
            convert_labels_to_json(
                image_dir=self._image_dir,
                label_dir=self._target_dir,
                output_json=self._target_json,
                fmt=self._fmt,
                remap=self._remap,
            )

    # --- Logging ---
    @override
    def _log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True

        console.rule(f"[bold red] {self._model}")
        console.log(f"[bold green]Model : {self._model}")
        console.log(f"[bold red]Data  : {self._data}")
        console.log(f"[bold]Device: {self._device}")

    @override
    def _log_results(self, results: dict[str, Any]):
        """Print the measured results."""
        pad = 10
        message = ""

        # Headers
        for m, v in results.items():
            if v:
                message += f"{f'{m}':<{pad}}\t"
        message += "\n"
        # Values
        for i, (m, v) in enumerate(results.items()):
            if v:
                if i == len(results) - 1:
                    message += f"{v:.{pad}f}\n"
                else:
                    message += f"{v:.{pad}f}\t"

        print(f"{message}\n")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
