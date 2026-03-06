#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference Runners.

This module provides several metric evaluators.
"""

from __future__ import annotations

__all__ = [
    "IQAEvaluator",
]

import argparse
import logging

import pyiqa
import pyiqa.default_model_configs
import pyiqa.models.inference_model
import torch
from box import Box

from mon.core import (
    console,
    create_progress_bar,
    DeviceLike,
    log_error,
    Path,
    PathLike,
    Size,
    SizeLike,
    sys_ctx,
)
from mon.dataset import DataLoader, IQADataset, transform as T

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class IQAEvaluator:
    """A runner for measuring IQA metrics."""

    METRICS = pyiqa.default_model_configs.DEFAULT_CONFIGS

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: PathLike,
        target_dir: PathLike | None,
        result_file: PathLike | None,
        arch: str,
        model: str,
        data: str,
        metric: list[str],
        device: DeviceLike = torch.device("cpu"),
        imgsz: SizeLike = 512,
        resize: bool = False,
        use_gt_mean: bool = False,
        save_txt: bool = False,
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
            metric (list[str]): List of metrics to measure.
            device (DeviceLike): Running device.
            imgsz (SizeLike, optional): Image size for resizing. If resize is
                False, this will be ignored. Defaults to 512.
            resize (bool, optional): Whether to resize images to ``imgsz``
                before measuring metrics. Defaults to False.
            use_gt_mean (bool, optional): Whether to use the mean of ground-truth
                images as the reference for NR metrics. Defaults to False.
            save_txt (bool, optional): Whether to save results in a text file.
                If False, results will only be printed to the console.
                Defaults to False.
            verbose (bool, optional): Verbose mode. Defaults to True.
        """
        # Assign attributes
        self.verbose = verbose
        self.device = sys_ctx.get_torch_device(device)
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.result_file = result_file
        self.arch = arch
        self.model = model
        self.data = data
        self.metric = metric
        self.imgsz = Size.from_value(imgsz)
        self.resize = resize
        self.use_gt_mean = use_gt_mean
        self.save_txt = save_txt

        # Allocate resources
        self._results = {}
        self._results_gt_mean = {}

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: PathLike):
        """Set the input directory."""
        self._input_dir = Path(input_dir).normalize()

    @property
    def target_dir(self) -> Path:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, target_dir: PathLike | None):
        """Set the target directory."""
        if target_dir:
            self._target_dir = Path(target_dir).normalize()
        else:
            self._target_dir = self._input_dir.replace_part("image", "target")

    @property
    def has_target(self) -> bool:
        """Check if the dataset has target data."""
        return self._target_dir is not None and self._target_dir.is_dir()

    @property
    def metric(self) -> list[str]:
        """Return the list of metrics."""
        return self._metric

    @metric.setter
    def metric(self, metric: list[str]):
        """Set the list of metrics."""
        self._metric = []
        self._metric_func = {}
        for i, m in enumerate(metric):
            if m in self.METRICS:
                self._metric.append(m)
                self._metric_func[m] = pyiqa.create_metric(
                    metric_name=m,
                    as_loss=False,
                    device=self.device
                )
            else:
                log_error(f"Unsupported metric: {m}. Skipping...")

    @property
    def metric_func(self) -> dict:
        """Return the dictionary of metric functions."""
        return self._metric_func

    @property
    def results(self) -> dict:
        """Return the dictionary of measured results."""
        return self._results

    @property
    def results_gt_mean(self) -> dict:
        """Return the dictionary of measured results with ground-truth mean."""
        return self._results_gt_mean

    # --- Creation ---
    @classmethod
    def from_config(cls, config: Box | dict) -> "IQAEvaluator":
        """Create an instance of IQAEvaluator from a configuration."""
        return cls(**config)

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
        parser.add_argument("--save-txt",    action="store_true")
        parser.add_argument("--verbose",     action="store_true")
        config = Box(vars(parser.parse_args()))
        return cls.from_config(config=config)

    # --- Measure ---
    def measure(self):
        """Run the metric measurement process."""
        # Summarize the current run
        self.log_summary()

        # Resolve dataloader
        # We don't persist the dataset to avoid memory consumption
        dataloader = self._build_dataloader()

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
        metric = self.metric
        metric_func = self.metric_func
        verbose = self.verbose

        # Prepare containers
        values = {m: [] for m in metric}
        results = {}

        # Processing loop
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
                target = datapoint.get("target")

                # Move tensors to a device
                image = image.to(device=device)
                target = target.to(device=device) if target is not None else None

                # Measure metric
                for m in metric:
                    if target is None and self.METRICS[m]["metric_mode"] == "FR":
                        continue
                    elif target is not None and self.METRICS[m]["metric_mode"] == "FR":
                        values[m].append(metric_func[m](image, target))
                    else:
                        values[m].append(metric_func[m](image))

        # Aggregate results
        for m, v in values.items():
            if len(v) > 0:
                results[m] = float(sum(v) / len(v))
            else:
                results[m] = None

        return results

    def _build_dataloader(self) -> DataLoader:
        """Create a dataloader for the given dataset."""
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])
        if self.resize:
            h, w = self.imgsz.hw
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
        results = self._results
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    iqa = IQAEvaluator.from_cli()
    iqa.measure()

# endregion
