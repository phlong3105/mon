#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rich Progress Bar Callback.

This module implements rich progress bar callback.
"""

from __future__ import annotations

__all__ = [
   "RichProgressBar",
]

import lightning
import torch
from lightning.pytorch.callbacks.progress import rich_progress

from mon.core import console, rich
from mon.globals import CALLBACKS


# region Rich Progress Bar

@CALLBACKS.register(name="rich_progress_bar")
class RichProgressBar(rich_progress.RichProgressBar):
    """The progress bar with rich text formatting."""
    
    def _init_progress(self, trainer: lightning.Trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console = console
            # self._console: Console = Console(**self._console_kwargs)
            self._console.clear_live()
            self._metric_component = rich_progress.MetricsTextColumn(
                trainer        = trainer,
                style          = self.theme.metrics,
                text_delimiter = self.theme.metrics_text_delimiter,
                metrics_format = self.theme.metrics_format,
            )
            self.progress = rich_progress.CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh = False,
                disable      = self.is_disabled,
                console      = self._console,
            )
            self.progress.start()
            # Progress has started
            self._progress_stopped = False
    
    def configure_columns(self, trainer: lightning.Trainer) -> list:
        if torch.cuda.is_available():
            return [
                rich.progress.TextColumn(
                    rich.console.get_datetime().strftime("[%m/%d/%Y %H:%M:%S]"),
                    justify = "left",
                    style   = "log.time"
                ),
                rich.progress.TextColumn("[progress.description][{task.description}]"),
                rich_progress.CustomBarColumn(
                    complete_style = self.theme.progress_bar,
                    finished_style = self.theme.progress_bar_finished,
                    pulse_style    = self.theme.progress_bar_pulse,
                ),
                rich_progress.BatchesProcessedColumn(style="progress.download"),
                "•",
                rich.GPUMemoryUsageColumn(devices=trainer.device_ids),
                "•",
                rich_progress.ProcessingSpeedColumn(style="progress.data.speed"),
                "•",
                rich.progress.TimeRemainingColumn(),
                ">",
                rich.progress.TimeElapsedColumn(),
                rich.progress.SpinnerColumn(),
            ]
        else:
            return [
                rich.progress.TextColumn(
                    rich.console.get_datetime().strftime("[%m/%d/%Y %H:%M:%S]"),
                    justify = "left",
                    style   = "log.time"
                ),
                rich.progress.TextColumn("[progress.description][{task.description}]"),
                rich_progress.CustomBarColumn(
                    complete_style = self.theme.progress_bar,
                    finished_style = self.theme.progress_bar_finished,
                    pulse_style    = self.theme.progress_bar_pulse,
                ),
                rich_progress.BatchesProcessedColumn(style="progress.download"),
                "•",
                rich_progress.ProcessingSpeedColumn(style="progress.data.speed"),
                "•",
                rich.progress.TimeRemainingColumn(),
                ">",
                rich.progress.TimeElapsedColumn(),
                rich.progress.SpinnerColumn(),
            ]

# endregion
