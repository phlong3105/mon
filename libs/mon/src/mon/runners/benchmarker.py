#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmarker.

This module provides several benchmarking tools.
"""

from __future__ import annotations

__all__ = [
    "Benchmarker",
]

import argparse
import logging
from typing import override

import torch

from mon.core import (
    console,
    create_progress_bar,
    DeviceLike,
    MODELS,
    Path,
    Size,
    SizeLike,
    sys_ctx,
    Task,
    TaskLike,
)
from mon.core.ui.prompt_toolkit import (
    ConfirmPrompt,
    IntPrompt,
    Prompt,
    PromptContextMixin,
)

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BENCHMARK
# ==============================================================================

class Benchmarker(PromptContextMixin):
    """A runner for benchmarking the performance of models."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        models: list[str],
        imgsz: SizeLike,
        num_runs: int,
        device: DeviceLike,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            models (list[str]): List of model fullnames to benchmark.
            imgsz (SizeLike): Image size for benchmarking.
            num_runs (int): Number of runs for latency measurement.
            device (DeviceLike): Running device.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__()

        # Assign attributes
        self.models = models
        self.imgsz = imgsz
        self.num_runs = num_runs
        self.device = device
        self.verbose = verbose

        # Allocate resources
        self.task = None
        self._results = {}  # Store all results and print them at the end of the run

    # --- Properties ---
    @property
    def task(self) -> Task | None:
        """Return the task type."""
        return self._task

    @task.setter
    def task(self, value: TaskLike | None):
        """Set the task type."""
        if value in Task:
            value = Task(value)
        self._task = value

    @property
    def models(self) -> list[str]:
        """Return the list of model fullnames to benchmark."""
        return self._models

    @models.setter
    def models(self, value: list[str]):
        """Set the list of model fullnames to benchmark."""
        all_models = MODELS.models
        value = value if isinstance(value, list) else list(value)
        value = [m for m in value if m in all_models]
        self._models = value

    @property
    def imgsz(self) -> Size:
        """Return the image size."""
        return self._imgsz

    @imgsz.setter
    def imgsz(self, value: SizeLike):
        """Set the image size."""
        self._imgsz = Size.from_value(value)

    @property
    def device(self) -> torch.device:
        """Return the device to use for computation."""
        return self._device

    @device.setter
    def device(self, value: DeviceLike):
        """Set the device to use for computation."""
        self._device = sys_ctx.get_torch_device(value)

    @property
    def results(self) -> dict[str, dict[str, float]]:
        """Return the dictionary of measured results."""
        return self._results

    # --- Creation ---
    @classmethod
    def from_cli(cls, task: TaskLike, **kwargs) -> "Benchmarker":
        """Create an instance of Benchmarker from command-line arguments."""
        parser = argparse.ArgumentParser(description="benchmark")
        parser.add_argument("--model",    type=str, action="append", help="Model fullnames.")
        parser.add_argument("--imgsz",    type=int, default=512)
        parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for latency measurement.")
        parser.add_argument("--device",   type=str, help="Running devices.")
        parser.add_argument("--verbose",  action="store_true")
        parser.add_argument("--prompt",   action="store_true", help="Prompt for additional inputs.")
        args = vars(parser.parse_args())

        args["models"] = args.pop("model")  # Rename "model" to "models"
        args |= kwargs  # Override with additional kwargs
        prompt = args.pop("prompt", False)

        obj = cls(**args)
        obj.task = task
        obj.prompt() if prompt else None
        return obj

    # --- Prompting ---
    @override
    def _display_prompt(self):
        """Display the current prompt."""
        if self._index == 0:
            # clear_terminal()
            console.rule(f"[bold red]Input Prompts")
        else:
            console.rule()

        if self._index == 0:
            # Task
            self.task = Prompt.ask(
                prompt="Task",
                choices=Task.values(),
                choices_repr=Task.values_repr(),
                defaults=self.task,
                strict=True,
            )
        if self._index == 1:
            # Models
            self.models = Prompt.ask(
                prompt="Models",
                choices=MODELS.search(task=self.task),
                defaults=None,
                multiselect=True,
                show_column=True,
                strict=True,
            )
        if self._index == 2:
            # Imgsz
            self.imgsz = IntPrompt.ask(
                prompt="Image Size",
                defaults=self.imgsz.h if self.imgsz else None,
            )
        if self._index == 3:
            # Num Runs
            self.num_runs = IntPrompt.ask(
                prompt="Number of Runs",
                defaults=self.num_runs,
            )
        if self._index == 4:
            # Device
            self.device = Prompt.ask(
                prompt="Device",
                choices=sys_ctx.device_names,
                defaults=sys_ctx.get_device(self.device).name,
                strict=True,
            )
        if self._index == 5:
            # Verbose
            self.verbose = ConfirmPrompt.ask(
                prompt="Verbose",
                defaults=self.verbose,
            )
        if self._index == self.num_prompts - 1:
            # Finish
            finish = ConfirmPrompt.ask(prompt="Finish/Re-input", defaults=True)
            if finish:
                self._index = self.num_prompts

    @override
    @property
    def num_prompts(self) -> int:
        """Return the total number of interactive steps."""
        return 6

    # --- Measure ---
    def measure(self):
        """Run the benchmark for the specified models."""
        # Summarize the current run
        self.log_summary()

        # Measuring
        self._results = self._measure()

        # Print results
        self.log_results()

    def _measure(self) -> dict[str, dict[str, float]]:
        """Measure the benchmark results for the specified models.

        Returns:
            dict[str, dict[str, float]]: A dictionary containing the benchmark
                results of each model.
        """
        results = {}

        with create_progress_bar(transient=True) as pbar:
            for i, m in pbar.track(
                sequence=enumerate(self.models),
                total=len(self.models),
                description="[bright_yellow]Benchmarking",
            ):
                # Define the model
                model = MODELS.build(name=m, device=self.device, verbose=False)
                model = model.to(self.device)
                model.eval()

                # Run benchmark
                stats = model.benchmark(imgsz=self.imgsz, num_runs=self.num_runs, verbose=False)
                results[m] = stats

        return results

    # --- Logging ---
    def log_summary(self):
        """Log a summary of the current run."""
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True
        console.rule(f"[bold red] Benchmark Models")
        console.log(f"[bold]Size  : {self.imgsz.hw}")
        console.log(f"[bold]Device: {self.device}\n")

    def log_results(self):
        """Print the measured results."""
        results = self.results
        pad = 10

        # Headers
        header = (
            f"{f'Model':<{pad}}\t"
            f"{f'Params (M)':<{pad}}\t"
            f"{f'MACs (G)':<{pad}}\t"
            f"{f'FLOPs (G)':<{pad}}\t"
            f"{f'Latency (ms)':<{pad}}\n"
        )
        header += "-" * ((pad + 4) * 6)

        # Rows
        message = ""
        for i, (model, stats) in enumerate(results.items()):
            message += (
                f"{f'{model}':<{pad}}\t"
                f"{self._format_unit(stats["params"], 'M'):<{pad}}\t"
                f"{self._format_unit(stats["macs"], 'G'):<{pad}}\t"
                f"{self._format_unit(stats["flops"], 'G'):<{pad}}\t"
                f"{f'{stats["latency"]:6.4f}':<{pad}}\n"
            )

        print(f"{header}")
        print(f"{message}")

    # --- Utilities ---
    # noinspection PyMethodMayBeStatic
    def _format_unit(self, value: float, target: str = "M") -> str:
        """Helper to format large numbers (e.g., 1.2G, 3.5M)."""
        if target == "G":
            return f"{value / 1e9:6.4f}"
        return f"{value / 1e6:6.4}"

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    benchmark_ = Benchmarker.from_cli(
        num_runs=10,
        device="cuda:0",
        verbose=True,
        prompt=True,
    )
    benchmark_.measure()

# endregion
